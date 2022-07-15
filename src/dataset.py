from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


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

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]

        color = directory / "color.png"
        img = read_image(str(color))

        if self.output_labels:
            label = self.labels.index(directory.parent.name)

            return img, label
