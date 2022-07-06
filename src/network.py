import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import IMG_SIZE


class ColorToDisp(nn.Module):
    def __init__(self):
        super(ColorToDisp, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, IMG_SIZE*IMG_SIZE*1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu(x)
        x = x.reshape((x.shape[0], 1, IMG_SIZE, IMG_SIZE))
        return x
