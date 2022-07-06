import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import IMG_SIZE

# Hyperparameters
START_CHANNELS = 4
CONV_LAYERS = 3
LRELU_ALPHA = 0.2


class Down(nn.Module):
    """
    Convolution 2D
    """

    def __init__(self, in_size, out_size):
        super(Down, self).__init__()

        self.conv = nn.Conv2d(in_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU(LRELU_ALPHA)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Up(nn.Module):
    """
    Convolution transpose 2D
    """

    def __init__(self, in_size, out_size):
        super(Up, self).__init__()

        self.conv = nn.ConvTranspose2d(in_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU(LRELU_ALPHA)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNPBRModel(nn.Module):
    def __init__(self):
        super(CNNPBRModel, self).__init__()

        downs = []
        ups = []

        for i in range(CONV_LAYERS):
            in_size = -1 if i == 0 else START_CHANNELS * 2 ** (i-1)
            out_size = START_CHANNELS * 2 ** i

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
