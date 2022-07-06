import os
import random

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

IMG_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


class TextureDataset(Dataset):
    def __init__(self, directory):
        self.dirs = [os.path.join(directory, f) for f in os.listdir(directory)]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]
        color = read_image(os.path.join(directory, "color.jpg"), mode=ImageReadMode.RGB)
        disp = read_image(os.path.join(directory, "disp.jpg"), mode=ImageReadMode.GRAY)
        color, disp = color.to(device), disp.to(device)
        color, disp = color / 255, disp / 255

        return color, disp


def get_dataloader(directory, batch_size):
    dataset = TextureDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

