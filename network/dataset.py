import os

import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from constants import *


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

        self.col_trans = torch.nn.Sequential(
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize([0, 0, 0], [255, 255, 255]),
        ).to(DEVICE)
        self.gray_trans = torch.nn.Sequential(
            self.col_trans,
            transforms.Grayscale(),
        ).to(DEVICE)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]

        color = self._read_img(directory, "color", True)
        normal = self._read_img(directory, "normal", True)
        disp = self._read_img(directory, "disp", False)
        rough = self._read_img(directory, "rough", False)

        out_data = torch.cat([normal, disp, rough], dim=0)

        return color, out_data

    def _read_img(self, dir, path, color: bool):
        path = os.path.join(dir, path+".jpg")
        img = read_image(path, ImageReadMode.RGB).to(DEVICE).float()
        if color:
            img = self.col_trans(img)
        else:
            img = self.gray_trans(img)

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
