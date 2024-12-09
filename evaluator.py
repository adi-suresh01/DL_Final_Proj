from typing import NamedTuple, List, Any
from dataclasses import dataclass
import torch
from tqdm.auto import tqdm
from schedulers import Scheduler
from models import Prober
from configs import ConfigBase
from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(self, device, model, probe_train_ds, probe_val_ds, config=default_config, quick_debug=False):
        self.device = device
        self.config = config
        self.model = model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        prober = Prober(repr_dim, self.config.prober_arch).to(self.device)
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), self.config.lr)
        scheduler = Scheduler(self.config.schedule, self.config.lr, dataset)

        for epoch in tqdm(range(self.config.epochs), desc="Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################
                # TODO: Modify for VICReg embeddings
                pred_encs, _ = model(states=batch.states, actions=batch.actions, return_targets=True)
                pred_encs = pred_encs.detach()
                target = self.normalizer.normalize_location(batch.locations)
                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                ################################################################
                losses = location_losses(pred_locs, target).mean()
                optimizer_pred_prober.zero_grad()
                losses.backward()
                optimizer_pred_prober.step()
                scheduler.step()

        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober, val_ds, prefix)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        probing_losses = []
        for batch in tqdm(val_ds, desc=f"Eval probe pred {prefix}"):
            pred_encs = self.model(states=batch.states).transpose(0, 1).detach()
            target = self.normalizer.normalize_location(batch.locations)
            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
            probing_losses.append(location_losses(pred_locs, target).cpu())
        return torch.stack(probing_losses, dim=0).mean().item()
