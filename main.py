from dataset import WallDataset, create_wall_dataloader, sequence_transforms
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from impl import JEPA, train_model
from torch.utils.data import DataLoader

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}
    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    device = get_device()
    model = JEPA(input_channels=2, hidden_dim=256, action_dim=2).to(device)
    #model = MockModel()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
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

    device = get_device()
    
    # Load training and validation datasets for probing (if needed)
    probe_train_ds, probe_val_ds = load_data(device)
    
    # Initialize the model
    model = load_model()
    model.apply(weights_init)
    
    # Create the training dataset (without normalization for now)
    training_dataset = WallDataset(
        data_path=f"/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=None  # Will set this after computing mean and std
    )
    

    # Create a DataLoader for the training dataset without shuffling
    temp_loader = DataLoader(
        training_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )
    
    # Initialize accumulators for sums and squared sums
    states_sum = 0.0
    states_squared_sum = 0.0
    actions_sum = 0.0
    actions_squared_sum = 0.0
    locations_sum = 0.0
    locations_squared_sum = 0.0
    num_states = 0
    num_actions = 0
    num_locations = 0

    for batch in temp_loader:
        states, actions, locations = batch.states, batch.actions, batch.locations  # Access batch data
        batch_size = states.size(0)
        seq_len = states.size(1)
        num_channels = states.size(2)
        height = states.size(3)
        width = states.size(4)
        
        # Reshape states to [batch_size * seq_len, channels, height, width]
        states_reshaped = states.view(-1, num_channels, height, width)
        num_pixels = states_reshaped.numel() / num_channels

        # Accumulate sums and squared sums for states
        states_sum += states_reshaped.sum(dim=(0, 2, 3))
        states_squared_sum += (states_reshaped ** 2).sum(dim=(0, 2, 3))
        num_states += num_pixels

        # Accumulate sums and squared sums for actions
        actions_sum += actions.sum(dim=(0, 1))
        actions_squared_sum += (actions ** 2).sum(dim=(0, 1))
        num_actions += actions.numel() / actions.size(-1)

        # Accumulate sums and squared sums for locations (if available)
        if locations.numel() != 0:
            locations_sum += locations.sum(dim=(0, 1))
            locations_squared_sum += (locations ** 2).sum(dim=(0, 1))
            num_locations += locations.numel() / locations.size(-1)

    # Compute mean and std for states
    states_mean = states_sum / num_states
    states_var = (states_squared_sum / num_states) - (states_mean ** 2)
    states_std = torch.sqrt(states_var)

    # Compute mean and std for actions
    actions_mean = actions_sum / num_actions
    actions_var = (actions_squared_sum / num_actions) - (actions_mean ** 2)
    actions_std = torch.sqrt(actions_var)

    # Compute mean and std for locations (if available)
    if num_locations > 0:
        locations_mean = locations_sum / num_locations
        locations_var = (locations_squared_sum / num_locations) - (locations_mean ** 2)
        locations_std = torch.sqrt(locations_var)
    else:
        locations_mean = None
        locations_std = None

    # Create normalization parameters dictionary
    normalization_params = {
        'states_mean': states_mean.view(1, -1, 1, 1),  # Shape [1, C, 1, 1]
        'states_std': states_std.view(1, -1, 1, 1),
        'actions_mean': actions_mean,
        'actions_std': actions_std,
        'locations_mean': locations_mean,
        'locations_std': locations_std
    }

    # Recreate the training dataset with normalization parameters
    training_dataset = WallDataset(
        data_path=f"/scratch/DL24FA/train",
        probing=False,
        device=device,
        transform=sequence_transforms,
        normalization_params=normalization_params
    )
    
    # Create the training DataLoader
    training_loader = DataLoader(
        training_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4
    )
    
    # Train the model
    train_model(model, training_loader, optimizer, num_epochs=10, device=device)
    
    # Evaluate the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
