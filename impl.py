import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# State Encoder
class StateEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(StateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 9 * 9, hidden_dim)
        self.bn = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn(self.fc(x)))
        return x


# Action Encoder
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super(ActionEncoder, self).__init__()
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.bn = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return F.relu(self.bn(self.fc1(x)))


# LSTM Temporal Model
class TemporalModel(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


# VICReg Loss Function
def vicreg_loss(predicted, target):
    mse_loss = F.mse_loss(predicted, target)

    # Variance Loss
    pred_std = predicted.std(dim=0) + 1e-4
    var_loss = torch.mean(F.relu(1 - pred_std))

    # Covariance Loss
    pred_centered = predicted - predicted.mean(dim=0, keepdim=True)
    cov_matrix = (pred_centered.T @ pred_centered) / (predicted.size(0) - 1)
    cov_loss = (cov_matrix - torch.diag(torch.diag(cov_matrix))).pow(2).sum()

    return mse_loss + 0.1 * var_loss + 0.1 * cov_loss


# Main JEPA Model
class JEPA(nn.Module):
    def __init__(self, state_channels, action_dim, hidden_dim):
        super(JEPA, self).__init__()
        self.state_encoder = StateEncoder(state_channels, hidden_dim)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        self.temporal_model = TemporalModel(hidden_dim)

    def forward(self, states, actions):
        if states.ndim != 5:
            raise ValueError(f"Expected states to have 5 dimensions, but got {states.ndim} dimensions.")
        batch_size, seq_len, _, _, _ = states.size()
        state_embeddings = []
        action_embeddings = []

        for t in range(seq_len):
            state_embeddings.append(self.state_encoder(states[:, t]))
            if t < seq_len - 1:
                action_embeddings.append(self.action_encoder(actions[:, t]))

        state_embeddings = torch.stack(state_embeddings, dim=1)
        action_embeddings = torch.stack(action_embeddings, dim=1)

        temporal_inputs = state_embeddings[:, :-1] + action_embeddings
        predicted_states = self.temporal_model(temporal_inputs)

        return predicted_states, state_embeddings[:, 1:]


def train_model(
    model, 
    dataloader, 
    optimizer, 
    val_loader=None,
    num_epochs=10,  
    device='cuda',  
    save_path=None
):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit='batch') as pbar:
            for states, locations, actions in dataloader:
                states, actions = states.to(device), actions.to(device)

                optimizer.zero_grad()
                predicted_states, target_states = model(states, actions)

                loss = vicreg_loss(predicted_states.flatten(0, 1), target_states.flatten(0, 1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'Train Loss': f'{loss.item():.6f}'})
                pbar.update(1)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {total_loss / len(dataloader):.6f}")
