import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize, ToTensor

IMG_SIZE = 256


class TextureDataset(Dataset):
    def __init__(self, directory):
        def dir_is_valid(d):
            if not os.path.isdir(d):
                return False
            files = os.listdir(d)
            return "color.jpg" in files and "disp.jpg" in files

        self.files = [os.path.join(directory, f) for f in os.listdir(directory)
            if dir_is_valid(os.path.join(directory, f))]

        self.resize = Resize((IMG_SIZE, IMG_SIZE))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        directory = self.files[idx]
        color = read_image(os.path.join(directory, "color.jpg"), mode=ImageReadMode.RGB)
        disp = read_image(os.path.join(directory, "disp.jpg"), mode=ImageReadMode.GRAY)
        color, disp = map(self.resize, (color, disp))
        color, disp = color.float() / 255, disp.float() / 255

        return color, disp


def get_dataloader(directory, batch_size=256):
    dataset = TextureDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

