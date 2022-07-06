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
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Up(nn.Module):
    """
    Convolve up.
    """

    def __init__(self, in_size, out_size):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(in_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ColorToDisp(nn.Module):
    """
    U-net from color map to displacement map.
    """

    def __init__(self):
        super(ColorToDisp, self).__init__()

        self.down1 = Down(3, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.down5 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        self.up5 = Up(16, 1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x
