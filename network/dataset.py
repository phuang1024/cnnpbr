import os
import random

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from constants import *


class TextureDataset(Dataset):
    def __init__(self, directory):
        self.dirs = [os.path.join(directory, f) for f in os.listdir(directory)]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]
        maps = {}
        for name in ("normal", "disp", "rough"):
            mode = ImageReadMode.RGB if name == "normal" else ImageReadMode.GRAY
            img = read_image(os.path.join(directory, f"{name}.jpg"), mode=mode)
            maps[name] = img

        data = combine_maps(maps)
        data = data.to(DEVICE)
        data = data / 255

        color = read_image(os.path.join(directory, "color.jpg"), mode=ImageReadMode.RGB)
        color = color.to(DEVICE)
        color = color / 255

        return color, data


def combine_maps(maps):
    """
    Joins channels of normal, disp, rough in that order.
    """
    size = maps["normal"].shape[1]
    data = torch.empty((3+1+1, size, size))
    data[0:3, :, :] = maps["normal"]
    data[3, :, :] = maps["disp"]
    data[4, :, :] = maps["rough"]

    return data

def extract_maps(data):
    """
    Reverse of combine_maps

    :param data: Numpy array shape (height, width, channels).
    """
    maps = {}
    maps["normal"] = data[:, :, 0:3]
    maps["disp"] = data[:, :, 3]
    maps["rough"] = data[:, :, 4]

    for name in maps:
        maps[name] = maps[name] * 255

    return maps


def get_dataloader(directory, batch_size):
    dataset = TextureDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

