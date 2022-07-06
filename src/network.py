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
        self.conv = nn.ConvTranspose2d(in_size, out_size, 3, padding=1)
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

    initial_ch = 16

    def __init__(self):
        super(ColorToDisp, self).__init__()

        ch = self.initial_ch

        self.down1 = Down(3, ch)
        self.down2 = Down(ch, ch*2)
        self.down3 = Down(ch*2, ch*4)
        self.down4 = Down(ch*4, ch*8)

        self.up1 = Up(ch, 1)
        self.up2 = Up(ch*2, ch)
        self.up3 = Up(ch*4, ch*2)
        self.up4 = Up(ch*8, ch*4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        #x = self.sigmoid(x)
        return x
