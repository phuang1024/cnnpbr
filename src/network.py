import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ColorToDisp(nn.Module):
    def __init__(self):
        super(ColorToDisp, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(1024*1024*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1024*1024*3),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu(x)
        return x
