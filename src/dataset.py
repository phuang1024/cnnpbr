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

    def __init__(self, noise_max):
        super().__init__()
        self.noise_max = noise_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rand = 2 * (torch.rand_like(x)-0.5)
        return x + rand * self.noise_max


class Transforms(nn.Module):
    """
    Transforms image data.
    Specify whether to augment in constructor.
    Input can be either color map or combined color/disp/... maps.
    Color adjustment will only be applied to color map (first three channels).
    """

    def __init__(self, augment: bool = False):
        super().__init__()
        self.augment = augment

        self.transforms = nn.Sequential(
            transforms.Resize(IMG_SIZE),
            transforms.Normalize(0, 255),
        )

        self.rand_trans = nn.Sequential(
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        )

        # Apply before normalizing
        self.color_uint = nn.Sequential(
            transforms.ColorJitter(0.08, 0.08, 0.08, 0.05),
            transforms.RandomAdjustSharpness(2),
        )

        # Apply after normalizing
        self.color_float = nn.Sequential(
            RandomNoise(0.02),
        )

    def forward(self, x):
        if self.augment:
            x[:3] = self.color_uint(x[:3])
        x = x.float()
        x = self.transforms(x)
        if self.augment:
            x = self.rand_trans(x)
            x[:3] = self.color_float(x[:3])
        x = torch.clamp(x, 0, 1)
        return x


class TextureDataset(Dataset):
    """
    Input: color map
    Output: disp map

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

    def __init__(self, data_path, augment: bool = True):
        super().__init__()
        self.transform = Transforms(augment)
        self.data_path = Path(data_path)
        self.dirs = []
        for d in self.data_path.iterdir():
            self.dirs.extend(d.iterdir())

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]
        color = read_image(str(directory / "color.png"), mode=ImageReadMode.RGB)
        disp = read_image(str(directory / "disp.png"), ImageReadMode.GRAY)

        data = torch.cat((color, disp), 0)
        data = self.transform(data)

        color = data[:3]
        disp = data[3:]
        return color, disp
