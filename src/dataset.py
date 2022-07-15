from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from constants import *


class Augmentation(nn.Module):
    """
    Augmentation of image data.
    Input can be either color map or combined color/disp/... maps.
    Color adjustment will only be applied to color map.
    """

    def __init__(self):
        super().__init__()

        self.transforms = nn.Sequential(
            transforms.Resize(IMG_SIZE),
            transforms.Normalize(128, 128),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        )

        self.color = nn.Sequential(
            transforms.ColorJitter(*AUG_JITTER),
            transforms.RandomAdjustSharpness(AUG_SHARP),
        )

    def forward(self, x):
        x = self.transforms(x)
        color = x[:3]
        color = self.color(color)
        x[:3] = color
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
        img = read_image(str(color)).float()

        if self.output_labels:
            img = self.augmentation(img)
            label = self.labels.index(directory.parent.name)
            return img, label