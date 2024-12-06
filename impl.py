import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class EncoderNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(EncoderNetwork, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully Connected Layer
        self.fc = nn.Linear(128 * 9 * 9, hidden_dim)
        self.fc_bn = nn.BatchNorm1d(hidden_dim)
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc_bn(self.fc(x))
        x = F.relu(x)
        x = self.projector(x)
        return x


class PredictorNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PredictorNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        # Projection Head
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class JEPA(nn.Module):

    def __init__(self, input_channels, hidden_dim, action_dim):
        super(JEPA, self).__init__()
        self.encoder = EncoderNetwork(input_channels, hidden_dim)
        self.predictor = PredictorNetwork(hidden_dim, action_dim)
        # Define the target encoder network (for BYOL)
        self.target_encoder = EncoderNetwork(input_channels, hidden_dim)
        # Initialize target encoder parameters
        self._initialize_target_encoder()
        # Set repr_dim to hidden_dim
        self.repr_dim = hidden_dim


    def _initialize_target_encoder(self):
        # Initialize target encoder parameters with encoder parameters
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(param.data)
        # Copy over the buffers (running_mean, running_var)
        for buffer, target_buffer in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target_buffer.data.copy_(buffer.data)


    def update_target_encoder(self, momentum=0.99):
        # Update target encoder parameters
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        # Update target encoder buffers
        for buffer, target_buffer in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target_buffer.data = buffer.data


    def forward(self, states, actions, return_targets=False):
        batch_size, seq_len, channels, height, width = states.size()  # Get the shape of the input states
        predicted_states = []
        target_states = []

        # Step 1: Encode the Initial Observation
        # Encode the first state to get the initial latent state
        s_t = self.encoder(states[:, 0])  # Encode the first time step
        predicted_states.append(s_t.unsqueeze(1))  # Add the latent state to predicted_states (unsqueeze to add time dimension)

        if return_targets:
            self.target_encoder.eval()
            with torch.no_grad():
                s_t_target = self.target_encoder(states[:, 0])
            target_states.append(s_t_target.unsqueeze(1))

        # Step 2: Recurrently Predict Future Latent States
        for t in range(1, seq_len):
            action_t = actions[:, t - 1]  # Get the action taken at time step t-1
            s_t = self.predictor(s_t, action_t)  # Predict the next latent state
            predicted_states.append(s_t.unsqueeze(1))  # Add the predicted latent state to the list

            if return_targets:
                s_t_target = self.target_encoder(states[:, t])
                target_states.append(s_t_target.unsqueeze(1))

        # Step 3: Concatenate Predicted States Across Time
        predicted_states = torch.cat(predicted_states, dim=1)  # Concatenate along the time dimension

        if return_targets:
            target_states = torch.cat(target_states, dim=1)
            return predicted_states, target_states
        else:
            return predicted_states

def compute_loss(predicted_states, target_states):
    # Normalize the representations
    predicted_states = F.normalize(predicted_states, dim=-1)
    target_states = F.normalize(target_states.detach(), dim=-1)  # Stop-gradient on target

    # Compute MSE loss
    loss = F.mse_loss(predicted_states, target_states)
    return loss

def train_model(model, dataloader, optimizer, num_epochs=10, momentum=0.99, device='cuda'):
    """
    Trains the JEPA model using the BYOL framework with tqdm progress bars and validations.
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Initialize tqdm progress bar for batches
        with tqdm(total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit='batch') as pbar:
            for states, _, actions in dataloader:
                states = states.to(device)  # Shape: [batch_size, seq_len, channels, height, width]
                actions = actions.to(device)  # Shape: [batch_size, seq_len - 1, action_dim]

                # **Validation 1: Verify Data Integrity**
                # Check shapes
                print(f"States shape: {states.shape}")
                print(f"Actions shape: {actions.shape}")
                # Check for NaNs or zeros
                assert not torch.isnan(states).any(), "NaNs detected in states."
                assert not torch.isnan(actions).any(), "NaNs detected in actions."
                assert states.abs().sum().item() != 0, "States tensor is all zeros."
                assert actions.abs().sum().item() != 0, "Actions tensor is all zeros."
                # Check data statistics
                print(f"States mean: {states.mean().item()}, std: {states.std().item()}")
                print(f"Actions mean: {actions.mean().item()}, std: {actions.std().item()}")

                optimizer.zero_grad()

                # Forward pass with return_targets=True to get both predicted and target states
                predicted_states, target_states = model(states, actions, return_targets=True)
                # predicted_states and target_states shape: [batch_size, seq_len, hidden_dim]

                # **Validation 2: Inspect Model Outputs**
                # Print mean and std of predicted and target states
                print(f"Predicted states mean: {predicted_states.mean().item()}, std: {predicted_states.std().item()}")
                print(f"Target states mean: {target_states.mean().item()}, std: {target_states.std().item()}")

                # Compute difference between predicted and target states
                difference = (predicted_states - target_states).abs()
                print(f"Difference mean: {difference.mean().item()}, max: {difference.max().item()}")

                # Flatten the representations to combine batch and sequence dimensions
                batch_size, seq_len, hidden_dim = predicted_states.size()
                predicted_states_flat = predicted_states.view(batch_size * seq_len, hidden_dim)
                target_states_flat = target_states.view(batch_size * seq_len, hidden_dim)

                # **Validation 3: Verify Loss Computation**
                # Compute loss across all time steps and batch instances
                loss = compute_loss(predicted_states_flat, target_states_flat)

                # Print loss value
                print(f"Loss before backward pass: {loss.item()}")

                # Check for NaNs or Infs in loss
                assert not torch.isnan(loss).any(), "NaNs detected in loss."
                assert not torch.isinf(loss).any(), "Infs detected in loss."
                assert loss.item() != 0, "Loss is zero."

                # Backward pass and optimization
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # **Validation 4: Confirm Gradients are Flowing**
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"Gradient norm for {name}: {grad_norm}")
                    else:
                        print(f"No gradient computed for {name}")

                optimizer.step()

                # Update the target encoder using exponential moving average
                model.update_target_encoder(momentum)

                total_loss += loss.item()

                # Update tqdm progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                pbar.update(1)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")

            # **Validation 5: Verify Target Encoder Updates**
            with torch.no_grad():
                diffs = []
                for param, target_param in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                    diffs.append((param - target_param).abs().mean().item())
                avg_diff = sum(diffs) / len(diffs)
                print(f"Average parameter difference between encoder and target encoder: {avg_diff}")
