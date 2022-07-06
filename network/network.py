import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import IMG_SIZE


class Down(nn.Module):
    """
    Convolve down.
    """

    def __init__(self, in_size, out_size):
        super(Down, self).__init__()

        self.conv = nn.Conv2d(in_size, out_size, 3, padding=1)
        #self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        return x


class Up(nn.Module):
    """
    Convolve up.
    """

    def __init__(self, in_size, out_size):
        super(Up, self).__init__()

        #self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.ConvTranspose2d(in_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ColorToDisp(nn.Module):
    """
    U-net from color map to displacement map.
    """

    channels = 4
    layers = 3

    def __init__(self):
        super(ColorToDisp, self).__init__()

        downs = []
        ups = []

        for i in range(self.layers):
            in_size = -1 if i == 0 else self.channels * 2 ** (i-1)
            out_size = self.channels * 2 ** i

            down = Down((3 if in_size == -1 else in_size), out_size)
            up = Up(out_size, (1 if in_size == -1 else in_size))
            downs.append(down)
            ups.append(up)

        self.downs = nn.Sequential(*downs)
        self.ups = nn.Sequential(*reversed(ups))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.downs(x)
        x = self.ups(x)
        x = self.sigmoid(x)
        return x
