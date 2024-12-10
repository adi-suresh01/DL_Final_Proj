from typing import NamedTuple, List, Any, Optional
from dataclasses import dataclass
import torch
from tqdm.auto import tqdm
from schedulers import Scheduler, LRSchedule
from models import Prober
from configs import ConfigBase
from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        Trains a prober to evaluate whether embeddings capture future locations.
        """
        repr_dim = self.model.temporal_model.lstm.hidden_size
        dataset = self.ds

        prober = Prober(repr_dim, self.config.prober_arch, output_shape=(2,)).to(self.device)
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), lr=self.config.lr)

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=self.config.lr,
            data_loader=dataset,
            epochs=self.config.epochs,
            optimizer=optimizer_pred_prober,
        )

        for epoch in tqdm(range(self.config.epochs), desc="Prober Training Epochs"):
            for batch in tqdm(dataset, desc="Prober Training Steps"):
                states, actions, locations = batch.states, batch.actions, batch.locations
                states, actions, locations = states.to(self.device), actions.to(self.device), locations.to(self.device)

                # TODO: Use JEPA to get state embeddings for probing
                state_embeddings = self.model.state_encoder(states[:, 0])
                pred_locs = prober(state_embeddings)

                # Normalize target locations
                target_locs = self.normalizer.normalize_location(locations)
                loss = torch.nn.functional.mse_loss(pred_locs, target_locs)

                optimizer_pred_prober.zero_grad()
                loss.backward()
                optimizer_pred_prober.step()

                scheduler.adjust_learning_rate()

        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        """
        Evaluates the prober on all validation datasets.
        """
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober=prober, val_ds=val_ds)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds):
        """
        Evaluates a trained prober on a validation dataset.
        """
        prober.eval()
        losses = []

        for batch in tqdm(val_ds, desc="Evaluating Prober"):
            states, actions, locations = batch.states, batch.actions, batch.locations
            states, actions, locations = states.to(self.device), actions.to(self.device), locations.to(self.device)

            # TODO: Use JEPA to get state embeddings for evaluation
            state_embeddings = self.model.state_encoder(states[:, 0])
            pred_locs = prober(state_embeddings)

            target_locs = self.normalizer.normalize_location(locations)
            loss = torch.nn.functional.mse_loss(pred_locs, target_locs)
            losses.append(loss.item())

        return sum(losses) / len(losses)
