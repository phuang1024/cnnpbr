import torch
from torch import nn

from constants import *


class NxConv(nn.Module):
    """
    Many convolution, batchnorm, leakyrelu layers.
    """

    def __init__(self, in_ch, out_ch, layer_count, kernel_size):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(layer_count):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(NET_LRELU_ALPHA),
            ))
            in_ch = out_ch

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Classifier(nn.Module):
    """
    Classifies color map.

    n * (conv, batchnorm, leakyrelu), linear, softmax
    """

    def __init__(self, num_labels: int):
        super().__init__()

        self.convs = nn.ModuleList()

        last_ch = None
        out_img_size = IMG_SIZE
        for i in range(NET_CLASS_LAYERS):
            in_ch = 3 if last_ch is None else last_ch
            out_ch = in_ch * 2
            out_img_size //= 2
            self.convs.append(nn.Sequential(
                NxConv(in_ch, out_ch, NET_CONV_LAYER_COUNT, NET_CONV_KERNEL_SIZE),
                nn.MaxPool2d(2),
            ))
            last_ch = out_ch

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(last_ch * out_img_size ** 2, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
