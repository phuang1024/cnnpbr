from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from constants import *


class RandomNoise(nn.Module):
    """
    Add random noise to each pixel.
    """

    def __init__(self, noise_max: float = 0.05):
        super().__init__()
        self.noise_max = noise_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rand = 2 * (torch.rand_like(x)-0.5)
        return x + rand * self.noise_max


class Augmentation(nn.Module):
    """
    Augmentation of image data.
    Input can be either color map or combined color/disp/... maps.
    Color adjustment will only be applied to color map (first three channels).
    """

    def __init__(self):
        super().__init__()

        self.transforms = nn.Sequential(
            transforms.Resize(IMG_SIZE),
            transforms.Normalize(128, 128),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        )

        # Apply before normalizing
        self.color_uint = nn.Sequential(
            transforms.ColorJitter(*AUG_JITTER),
            transforms.RandomAdjustSharpness(AUG_SHARP),
        )

        # Apply after normalizing
        self.color_float = nn.Sequential(
            RandomNoise(AUG_NOISE),
        )

    def forward(self, x):
        x[:3] = self.color_uint(x[:3])
        x = x.float()
        x = self.transforms(x)
        x[:3] = self.color_float(x[:3])
        return x


class TextureDataset(Dataset):
    """
    Input: color map
    Output: label OR disp map (specify in constructor)

    Make your data path structure like this:

    data_path
    |__ label1
    |   |__ texture1
    |   |   |__ color.png
    |   |   |__ ao.png
    |   |   |   ...
    |   |   ...
    |   ...
    """

    def __init__(self, data_path, output_labels: bool):
        self.output_labels = output_labels
        self.data_path = Path(data_path)
        self.labels = sorted(p.name for p in self.data_path.iterdir() if p.is_dir())

        self.dirs = []
        for d in self.data_path.iterdir():
            self.dirs.extend(d.iterdir())

        self.augmentation = Augmentation()

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]

        color = directory / "color.png"
        color = read_image(str(color))

        if self.output_labels:
            color = self.augmentation(color)
            label = self.labels.index(directory.parent.name)
            return color, label

        else:
            disp = directory / "disp.png"
            disp = read_image(str(disp), ImageReadMode.GRAY)

            data = torch.cat((color, disp), 0)
            data = self.augmentation(data)
            color = data[:3]
            disp = data[3:]
            return color, disp
