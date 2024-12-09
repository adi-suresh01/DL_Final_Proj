import torch
import torch.nn as nn
import os
import pickle
from dataset import WallDataset, sequence_transforms
from evaluator import ProbingEvaluator
from impl import JEPA, train_model
from torch.utils.data import DataLoader


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def compute_normalization(loader, device):
    """Compute normalization parameters for states, actions, and locations."""
    states_sum, states_squared_sum = 0.0, 0.0
    actions_sum, actions_squared_sum = 0.0, 0.0
    locations_sum, locations_squared_sum = 0.0, 0.0
    num_states, num_actions, num_locations = 0, 0, 0

    for batch in loader:
        states, actions, locations = batch.states, batch.actions, batch.locations
        states, actions = states.to(device), actions.to(device)

        # Compute stats for states
        states_reshaped = states.view(states.size(0) * states.size(1), states.size(2), states.size(3), states.size(4))
        states_sum += states_reshaped.sum(dim=(0, 2, 3))
        states_squared_sum += (states_reshaped ** 2).sum(dim=(0, 2, 3))
        num_states += states_reshaped.numel() / states_reshaped.size(1)

        # Compute stats for actions
        actions_sum += actions.sum(dim=(0, 1))
        actions_squared_sum += (actions ** 2).sum(dim=(0, 1))
        num_actions += actions.numel() / actions.size(-1)

        # Compute stats for locations (if available)
        if locations.numel() > 0:
            locations_sum += locations.sum(dim=(0, 1))
            locations_squared_sum += (locations ** 2).sum(dim=(0, 1))
            num_locations += locations.numel() / locations.size(-1)

    # Calculate means and stds
    states_mean = states_sum / num_states
    states_std = torch.sqrt((states_squared_sum / num_states) - (states_mean ** 2))
    actions_mean = actions_sum / num_actions
    actions_std = torch.sqrt((actions_squared_sum / num_actions) - (actions_mean ** 2))

    locations_mean, locations_std = None, None
    if num_locations > 0:
        locations_mean = locations_sum / num_locations
        locations_std = torch.sqrt((locations_squared_sum / num_locations) - (locations_mean ** 2))

    return {
        'states_mean': states_mean.view(1, -1, 1, 1),
        'states_std': states_std.view(1, -1, 1, 1),
        'actions_mean': actions_mean,
        'actions_std': actions_std,
        'locations_mean': locations_mean,
        'locations_std': locations_std
    }


# def save_normalization_params(normalization_params, path):
#     with open(path, 'wb') as f:
#         pickle.dump(normalization_params, f)


# def load_normalization_params(path):
#     if os.path.exists(path):
#         with open(path, 'rb') as f:
#             return pickle.load(f)
#     return None


# def load_model(path, model, optimizer=None, device='cuda'):
#     checkpoint = torch.load(path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     if optimizer:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     print(f"Model loaded from {path}")
#     return model, checkpoint['epoch']


def create_dataset(data_path, probing, device, transform, normalization_params=None, batch_size=64, shuffle=True):
    dataset = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        transform=transform,
        normalization_params=normalization_params
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=not probing
    )
    return dataset, dataloader


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """Evaluate the model using ProbingEvaluator."""
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


def weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Set device and paths
    device = get_device()
    save_path = "/scratch/as17339/jepa_model_vicreg.pth"
    norm_path = "/scratch/as17339/normalization_params_vicreg.pkl"

    # Initialize model
    model = JEPA(input_channels=2, hidden_dim=256, action_dim=2).to(device)
    model.apply(weights_init)

    # Load or compute normalization parameters
    # normalization_params = load_normalization_params(norm_path)
    temp_dataset, temp_loader = create_dataset(
        data_path=f"/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=None,
        batch_size=64,
        shuffle=False
    )
    
    normalization_params = compute_normalization(temp_loader, device)
        # save_normalization_params(normalization_params, norm_path)

    # Create datasets and dataloaders
    training_dataset, training_loader = create_dataset(
        data_path=f"/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=normalization_params
    )
    val_dataset, val_loader = create_dataset(
        data_path=f"/scratch/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        transform=sequence_transforms,
        normalization_params=normalization_params
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4
    )

    # Check if a saved model exists
    # try:
    #     model, start_epoch = load_model(save_path, model, optimizer, device=device)
    # except FileNotFoundError:
    #     print("No pre-trained model found. Training a new model.")
    train_model(
        model=model,
        dataloader=training_loader,
        optimizer=optimizer,
        val_loader=val_loader,
        num_epochs=10,
        device=device,
        save_path=save_path,
        momentum=0.99
    )

    # Evaluate the model
    probe_train_ds, probe_val_ds = val_dataset, val_loader
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
