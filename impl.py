import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# SE Block with configurable reduction factor
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()  # Batch size, Channels, Height, Width
        y = self.global_avg_pool(x).view(b, c)  # Global average pooling
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y  # Scale the input feature map


# Encoder Network with SE Blocks and Swish activation
class EncoderNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(EncoderNetwork, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)
        
        # Fully Connected Layer
        self.fc = nn.Linear(128 * 9 * 9, hidden_dim)
        self.fc_ln = nn.LayerNorm(hidden_dim)
        
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))  # Swish activation
        x = self.se1(x)
        
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        
        x = F.silu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc_ln(self.fc(x))
        x = F.silu(x)
        x = self.projector(x)
        return x


# Predictor Network with Action Embedding and Residual Connections
class PredictorNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PredictorNetwork, self).__init__()
        # Action Embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim // 2)
        
        # MLP Layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Residual Connection
        self.residual = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

    def forward(self, state, action):
        action_embedded = self.action_embed(action)  # Embed action
        x = torch.cat([state, action_embedded], dim=-1)
        residual = self.residual(x)  # Residual connection
        x = self.mlp(x)
        return x + residual  # Add residual connection


# JEPA Model with Symmetrized Loss
class JEPA(nn.Module):
    def __init__(self, input_channels, hidden_dim, action_dim):
        super(JEPA, self).__init__()
        self.encoder = EncoderNetwork(input_channels, hidden_dim)
        self.predictor = PredictorNetwork(hidden_dim, action_dim)
        self.target_encoder = EncoderNetwork(input_channels, hidden_dim)
        self._initialize_target_encoder()
        self.repr_dim = hidden_dim

    def _initialize_target_encoder(self):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(param.data)
        for buffer, target_buffer in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target_buffer.data.copy_(buffer.data)

    def update_target_encoder(self, momentum=0.99):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        for buffer, target_buffer in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target_buffer.data = buffer.data

    def forward(self, states, actions, return_targets=False):
        batch_size, seq_len, _, _, _ = states.size()
        predicted_states = []
        target_states = []

        s_t = self.encoder(states[:, 0])
        predicted_states.append(s_t.unsqueeze(1))

        if return_targets:
            self.target_encoder.eval()
            with torch.no_grad():
                s_t_target = self.target_encoder(states[:, 0])
            target_states.append(s_t_target.unsqueeze(1))

        for t in range(1, seq_len):
            action_t = actions[:, t - 1]
            s_t = self.predictor(s_t, action_t)
            predicted_states.append(s_t.unsqueeze(1))

            if return_targets:
                s_t_target = self.target_encoder(states[:, t])
                target_states.append(s_t_target.unsqueeze(1))

        predicted_states = torch.cat(predicted_states, dim=1)

        if return_targets:
            target_states = torch.cat(target_states, dim=1)
            return predicted_states, target_states
        else:
            return predicted_states


# Symmetrized Loss Function
def compute_loss(predicted_states, target_states):
    predicted_states = F.normalize(predicted_states, dim=-1)
    target_states = F.normalize(target_states.detach(), dim=-1)

    mse_loss = F.mse_loss(predicted_states, target_states)

    pred_std = predicted_states.std(dim=0) + 1e-4
    var_loss = torch.mean(F.relu(1 - pred_std))

    pred_centered = predicted_states - predicted_states.mean(dim=0, keepdim=True)
    cov_matrix = (pred_centered.T @ pred_centered) / (predicted_states.size(0) - 1)
    cov_loss = (cov_matrix - torch.diag(torch.diag(cov_matrix))).pow(2).sum() / predicted_states.size(1)

    loss = mse_loss + 0.1 * var_loss + 0.1 * cov_loss
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
