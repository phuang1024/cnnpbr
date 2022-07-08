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

        color = self._read_img(directory, "color")
        normal = self._read_img(directory, "normal")
        disp = self._read_img(directory, "disp")
        rough = self._read_img(directory, "rough")

        out_data = torch.cat([normal, disp, rough], dim=0)

        return color, out_data

    @staticmethod
    def _read_img(dir, path):
        path = os.path.join(dir, path+".jpg")
        img = read_image(path).float()
        img = img / 255.0
        return img


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
