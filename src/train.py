import os

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    dataloader = get_dataloader("data")

    model = ColorToDisp().to(device)
    print(model)

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "model.pth")
