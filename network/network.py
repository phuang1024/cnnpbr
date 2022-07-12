import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import IMG_SIZE


class ConvDown(nn.Module):
    """
    2x (conv, batchnorm, leakyrelu)
    """

    def __init__(self, in_channels, out_channels, alpha=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class ConvUp(nn.Module):
    """
    conv transpose, batchnorm, leakyrelu
    """

    def __init__(self, in_channels, out_channels, alpha=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class CNNPBRModel(nn.Module):
    """
    U-net.
    """

    def __init__(self, layers=4, alpha=0.2):
        super().__init__()

        self.layers = layers

        # Left, down layers
        for i in range(layers):
            in_channels = 3 if i == 0 else getattr(self, f"down{i-1}").out_channels
            out_channels = 2 ** (i+2)

            conv = ConvDown(in_channels, out_channels, alpha)
            setattr(self, f"down{i}", conv)

            if i != layers-1:
                maxpool = nn.MaxPool2d(2)
                setattr(self, f"maxpool{i}", maxpool)

        # Right, up layers
        for i in range(layers-1):
            in_channels = getattr(self, f"down{layers-1}").out_channels if i == 0 else \
                getattr(self, f"up{i-1}").out_channels

            upsamp = nn.Upsample(scale_factor=2, mode="bilinear")
            setattr(self, f"upsamp{i}", upsamp)

            # Concatenates upsamp with corresponding down layer
            real_in_channels = getattr(self, f"down{layers-i-2}").out_channels + in_channels
            out_channels = real_in_channels // 2
            conv = ConvUp(real_in_channels, out_channels, alpha)
            setattr(self, f"up{i}", conv)

        # Regression (output)
        in_channels = getattr(self, f"up{layers-2}").out_channels
        self.reg_nrmr = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_nrmg = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_nrmb = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_disp = nn.Conv2d(in_channels, 1, 1)
        self.reg_rough = nn.Conv2d(in_channels, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Left, down layers
        lefts = []
        for i in range(self.layers):
            x = getattr(self, f"down{i}")(x)
            lefts.append(x)
            if i != self.layers-1:
                x = getattr(self, f"maxpool{i}")(x)

        # Right, up layers
        for i in range(self.layers-1):
            x = getattr(self, f"upsamp{i}")(x)
            x = torch.cat([x, lefts[self.layers-i-2]], dim=1)
            x = getattr(self, f"up{i}")(x)

        nrmr = self.reg_nrmr(x)
        nrmg = self.reg_nrmg(x)
        nrmb = self.reg_nrmb(x)
        disp = self.reg_disp(x)
        rough = self.reg_rough(x)
        final = torch.cat([nrmr, nrmg, nrmb, disp, rough], dim=1)

        final = self.sigmoid(final)
        return final
