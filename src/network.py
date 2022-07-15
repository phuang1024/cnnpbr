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


class Network(nn.Module):
    """
    Generates displacement map from color map.
    Many levels of NxConv, MaxPool which will be put together to form the final image.

    Input                            ______
    |_ conv0 ---------------------> |      |
       |_ maxpool1                  |      |
       |_ conv1 ------- upsamp ---> | conv | --> displacement map
          |_ maxpool2               |      |
          |_ conv2 ---- upsamp ---> |      |
             ...
    """

    def __init__(self):
        super().__init__()

        for layer in range(NET_LAYERS):
            in_ch = 3 if layer == 0 else NET_CONV_CHANNELS
            conv = NxConv(in_ch, NET_CONV_CHANNELS, NET_CONV_LAYERS, NET_CONV_KERNEL)
            self.add_module(f"conv{layer}", conv)
            if layer > 0:
                pool = nn.MaxPool2d(2, 2)
                self.add_module(f"pool{layer}", pool)
                upsamp = nn.Upsample(scale_factor=2 ** layer, mode="bilinear")
                self.add_module(f"upsamp{layer}", upsamp)

        self.regression = nn.Conv2d(NET_CONV_CHANNELS * NET_LAYERS, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        outputs = []
        for layer in range(NET_LAYERS):
            if layer == 0:
                x = self._modules[f"conv{layer}"](x)
                outputs.append(x)
            else:
                x = self._modules[f"pool{layer}"](x)
                x = self._modules[f"conv{layer}"](x)
                out = self._modules[f"upsamp{layer}"](x)
                outputs.append(out)

        data = torch.cat(outputs, 1)
        data = self.regression(data)
        data = self.tanh(data)
        return data
