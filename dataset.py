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

        sample = {"states": states, "actions": actions, "locations": locations}

        if self.transform:
            sample = self.transform(sample)
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
        img_sequence = [
            TF.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness)) for img in img_sequence
        ]
        img_sequence = [
            TF.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast)) for img in img_sequence
        ]
        return {"states": img_sequence, "actions": actions, "locations": locations}


class RandomCropSequence(object):
    """Randomly crops a sequence of images."""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_sequence, actions, locations = sample["states"], sample["actions"], sample["locations"]
        img_sequence = [TF.center_crop(img, self.crop_size) for img in img_sequence]
        return {"states": img_sequence, "actions": actions, "locations": locations}


class RandomHorizontalFlipSequence(object):
    """Randomly horizontally flips the sequence of images and adjusts locations and actions."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img_sequence, actions, locations = sample["states"], sample["actions"], sample["locations"]
        if random.random() < self.p:
            img_sequence = [TF.hflip(img) for img in img_sequence]
            image_width = img_sequence[0].shape[-1]
            if locations.numel() != 0:
                locations = adjust_locations_horizontal_flip(locations, image_width)
            actions = adjust_actions_horizontal_flip(actions)
        return {"states": img_sequence, "actions": actions, "locations": locations}


class RandomRotationSequence(object):
    """Randomly rotates the sequence of images and adjusts locations and actions."""
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, sample):
        img_sequence, actions, locations = sample["states"], sample["actions"], sample["locations"]
        angle = random.uniform(self.degrees[0], self.degrees[1])
        img_sequence = [TF.rotate(img, angle, expand=False) for img in img_sequence]
        image_size = img_sequence[0].shape  # [C, H, W]
        if locations.numel() != 0:
            locations = adjust_locations_rotation(locations, angle, image_size)
        actions = adjust_actions_rotation(actions, angle)
        return {"states": img_sequence, "actions": actions, "locations": locations}


def adjust_locations_horizontal_flip(locations, image_width):
    locations = locations.clone()
    locations[..., 0] = image_width - locations[..., 0]
    return locations


def adjust_locations_rotation(locations, angle_degrees, image_size):
    angle_radians = -angle_degrees * (math.pi / 180.0)
    device = locations.device
    rotation_matrix = torch.tensor(
        [
            [math.cos(angle_radians), -math.sin(angle_radians)],
            [math.sin(angle_radians), math.cos(angle_radians)],
        ],
        dtype=locations.dtype,
        device=device,
    )
    image_center = torch.tensor([image_size[-1] / 2.0, image_size[-2] / 2.0], dtype=locations.dtype, device=device)
    locations = locations - image_center
    locations = torch.matmul(locations, rotation_matrix)
    locations = locations + image_center
    return locations


def adjust_actions_horizontal_flip(actions):
    actions = actions.clone()
    actions[..., 0] = -actions[..., 0]
    return actions


def adjust_actions_rotation(actions, angle_degrees):
    angle_radians = -angle_degrees * (math.pi / 180.0)
    device = actions.device
    rotation_matrix = torch.tensor(
        [
            [math.cos(angle_radians), -math.sin(angle_radians)],
            [math.sin(angle_radians), math.cos(angle_radians)],
        ],
        dtype=actions.dtype,
        device=device,
    )
    actions = torch.matmul(actions, rotation_matrix)
    return actions


sequence_transforms = transforms.Compose([
    RandomHorizontalFlipSequence(p=0.5),
    RandomRotationSequence(degrees=10),
    RandomBrightnessContrastSequence(brightness=0.3, contrast=0.3),
    # Add more transformations here if needed
])
