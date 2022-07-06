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

        self._preprocess()

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        directory = self.dirs[idx]
        color = read_image(os.path.join(directory, "color.jpg"), mode=ImageReadMode.RGB)
        disp = read_image(os.path.join(directory, "disp.jpg"), mode=ImageReadMode.GRAY)
        color, disp = color.to(device), disp.to(device)
        color, disp = color / 255, disp / 255

        return color, disp

    def _preprocess(self):
        """
        Resize all images to desired size and store in tmp dir.
        """
        tmpdir = os.path.join("/tmp", f"cnnpbr{random.randint(0, 1e9)}")
        os.makedirs(tmpdir, exist_ok=True)

        new_dirs = []
        for dir in tqdm(self.dirs, desc="Preparing data"):
            new = os.path.join(tmpdir, os.path.basename(dir))
            new_dirs.append(new)
            os.makedirs(new, exist_ok=True)
            for f in os.listdir(dir):
                if f.endswith(".jpg"):
                    read_mode = cv2.IMREAD_COLOR if "color" in f.lower() or "normal" in f.lower() \
                        else cv2.IMREAD_GRAYSCALE
                    img = cv2.imread(os.path.join(dir, f), read_mode)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(os.path.join(new, f), img)

        self.dirs = new_dirs


def get_dataloader(directory, batch_size):
    dataset = TextureDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

