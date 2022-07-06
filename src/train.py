import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class TextureDataset(Dataset):
    def __init__(self, directory, transform=None):
        def dir_is_valid(directory):
            if not os.path.isdir(directory):
                return False
            files = os.listdir(directory)
            return "color.jpg" in files and "disp.jpg" in files

        self.directory = directory
        self.transform = transform
        self.files = list(filter(dir_is_valid, os.listdir(directory)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        directory = os.path.join(self.directory, self.files[idx])
        color = read_image(os.path.join(directory, "color.jpg"))
        disp = read_image(os.path.join(directory, "disp.jpg"))

        if self.transform:
            color = self.transform(color)
            disp = self.transform(disp)

        return color, disp


def get_dataloader(directory, batch_size=64):
    dataset = TextureDataset(directory, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train():
    dataloader = get_dataloader("data")


if __name__ == "__main__":
    train()
