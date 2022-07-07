import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import IMG_SIZE

# Hyperparameters
CONV_LAYERS = 5
LRELU_ALPHA = 0.2


class Down(nn.Module):
    """
    Convolution 2D and MaxPooling 2D
    """

    def __init__(self, in_size, out_size):
        super(Down, self).__init__()

        self.conv = nn.Conv2d(in_size, out_size, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU(LRELU_ALPHA)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Up(nn.Module):
    """
    Upsample 2D and Convolution 2D transpose
    """

    def __init__(self, in_size, out_size):
        super(Up, self).__init__()

        self.conv = nn.ConvTranspose2d(in_size, out_size, 3, padding=1)
        self.upsamp = nn.Upsample(scale_factor=2, mode="bilinear")
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU(LRELU_ALPHA)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsamp(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNPBRModel(nn.Module):
    def __init__(self):
        super(CNNPBRModel, self).__init__()

        downs = []
        ups = []

        # Down layers
        down_sizes = []
        for i in range(CONV_LAYERS+1):
            size = 3 if i == 0 else 2 ** (i+1)
            down_sizes.append(size)

        for i in range(CONV_LAYERS):
            layer = Down(down_sizes[i], down_sizes[i+1])
            setattr(self, f"down{i}", layer)

        # Up layers
        up_sizes = []  # list of (layer_output_depth, after_concat_depth)
        for i in range(CONV_LAYERS+1):
            if i == 0:
                size = 2 ** (CONV_LAYERS+1)
            elif i == CONV_LAYERS:
                size = up_sizes[-1][1] // 2
            else:
                output_size = up_sizes[-1][1] // 2
                real_size = output_size + down_sizes[CONV_LAYERS-i]
                up_sizes.append((output_size, real_size))
                continue

            up_sizes.append((size, size))

        for i in range(CONV_LAYERS):
            layer = Up(up_sizes[i][1], up_sizes[i+1][0])
            setattr(self, f"up{i}", layer)

        self.regression = nn.Conv2d(up_sizes[-1][1], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for i in range(CONV_LAYERS):
            down = getattr(self, f"down{i}")
            x = down(x)
            outputs.append(x)

        for i in range(CONV_LAYERS):
            up = getattr(self, f"up{i}")
            x = up(x)
            if i != CONV_LAYERS - 1:
                # Dimensions are (batch, channels, width, height)
                x = torch.concat((x, outputs[CONV_LAYERS-i-2]), dim=1)

        x = self.regression(x)
        x = self.sigmoid(x)

        return x
