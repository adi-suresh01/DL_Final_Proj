import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import math
from torchvision import transforms
from typing import NamedTuple, Optional
import numbers


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        transform=None,
        normalization_params=None,
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        self.transform = transform
        self.normalization_params = normalization_params
        if normalization_params is not None:
            self.normalization_params = {
                key: val.to(self.device) if val is not None else None
                for key, val in normalization_params.items()
            }

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i].copy()).float().to(self.device)
        actions = torch.from_numpy(self.actions[i].copy()).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i].copy()).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        # Add a sequence dimension if it doesn't exist
        if states.ndim == 3:  # If states are [C, H, W] instead of [Seq, C, H, W]
            states = states.unsqueeze(0)

        sample = {"states": [states], "actions": actions, "locations": locations}

        if self.transform:
            sample = self.transform(sample)

        # Ensure `states` is a list of tensors
        if isinstance(sample["states"], torch.Tensor):  # If transform didn't create a list
            sample["states"] = [sample["states"]]

        states = torch.stack(sample["states"])  # Shape: [seq_len, channels, height, width]
        actions = sample["actions"]
        locations = sample["locations"]

        if self.normalization_params is not None:
            states = (states - self.normalization_params["states_mean"]) / self.normalization_params["states_std"]
            actions = (actions - self.normalization_params["actions_mean"]) / self.normalization_params["actions_std"]

            if self.locations is not None:
                locations = (locations - self.normalization_params["locations_mean"]) / self.normalization_params["locations_std"]

        return WallSample(states=states, locations=locations, actions=actions)


class RandomBrightnessContrastSequence(object):
    """Randomly adjusts brightness and contrast for a sequence of images."""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        img_sequence, actions, locations = sample["states"], sample["actions"], sample["locations"]

        if img_sequence[0].shape[0] not in [1, 3]:  # Only apply to grayscale or RGB
            return {"states": img_sequence, "actions": actions, "locations": locations}

        img_sequence = [
            TF.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness)) for img in img_sequence
        ]
        img_sequence = [
            TF.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast)) for img in img_sequence
        ]
        return {"states": img_sequence, "actions": actions, "locations": locations}


sequence_transforms = transforms.Compose([
    RandomBrightnessContrastSequence(brightness=0.3, contrast=0.3),
])
