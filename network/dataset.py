import os
import random

import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from constants import *


class RandomRot90x(torch.nn.Module):
    """
    Randomly rotate 90x with equal probabilities.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        :param img: torch.Tensor
        :return: torch.Tensor
        """
        return torch.rot90(img, random.randint(0, 3))


class TextureDataset(Dataset):
    def __init__(self, directory):
        def validate_dir(dir):
            if not os.path.isdir(dir):
                return False
            for name in ("color", "normal", "disp", "rough"):
                if not os.path.isfile(os.path.join(dir, name + ".jpg")):
                    return False
            return True

        self.dirs = [os.path.join(directory, f) for f in os.listdir(directory)
            if validate_dir(os.path.join(directory, f))]

        self.all_trans = torch.nn.Sequential(
            transforms.Normalize(0, 255),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        ).to(DEVICE)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]

        color = self._read_img(directory, "color")
        normal = self._read_img(directory, "normal")
        disp = self._read_img(directory, "disp")
        rough = self._read_img(directory, "rough")

        all_data = torch.cat([color, normal, disp, rough], dim=0).to(DEVICE)
        all_data = all_data.float()
        all_data = self.all_trans(all_data)
        in_data = all_data[:3, ...]
        out_data = all_data[3:, ...]

        return in_data, out_data

    def _read_img(self, dir, name):
        path = os.path.join(dir, name+".jpg")
        mode = ImageReadMode.RGB if name in ("color", "normal") else ImageReadMode.GRAY
        img = read_image(path, mode)
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
