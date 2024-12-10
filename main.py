import torch
from dataset import WallDataset, sequence_transforms
from evaluator import ProbingEvaluator
from impl import JEPA, train_model, vicreg_loss
from torch.utils.data import DataLoader


def get_device():
    """Check for GPU availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_normalization(loader, device):
    stats = {
        "states_sum": 0.0,
        "states_squared_sum": 0.0,
        "actions_sum": 0.0,
        "actions_squared_sum": 0.0,
        "num_states": 0,
        "num_actions": 0,
    }

    for batch in loader:
        states, actions, _ = batch.states.to(device), batch.actions.to(device), batch.locations
        stats["states_sum"] += states.sum()
        stats["states_squared_sum"] += (states ** 2).sum()
        stats["actions_sum"] += actions.sum()
        stats["actions_squared_sum"] += (actions ** 2).sum()
        stats["num_states"] += states.numel()
        stats["num_actions"] += actions.numel()

    return {
        "states_mean": stats["states_sum"] / stats["num_states"],
        "states_std": torch.sqrt(stats["states_squared_sum"] / stats["num_states"]),
        "actions_mean": stats["actions_sum"] / stats["num_actions"],
        "actions_std": torch.sqrt(stats["actions_squared_sum"] / stats["num_actions"]),
    }


def create_dataset(data_path, probing, device, transform, normalization_params=None, batch_size=64, shuffle=True):
    dataset = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        transform=transform,
        normalization_params=normalization_params,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=not probing,
    )
    return dataset, dataloader


if __name__ == "__main__":
    device = get_device()

    # Model Initialization
    model = JEPA(state_channels=2, action_dim=2, hidden_dim=256).to(device)

    # Compute normalization parameters
    temp_ds, temp_loader = create_dataset(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=None,
    )
    normalization_params = compute_normalization(temp_loader, device)

    # Dataset and Dataloader creation
    train_ds, train_loader = create_dataset(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=normalization_params,
    )
    val_ds, val_loader = create_dataset(
        data_path="/scratch/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        transform=sequence_transforms,
        normalization_params=normalization_params,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    train_model(
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
    )
