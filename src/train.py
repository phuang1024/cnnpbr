import os

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader


def train():
    dataloader = get_dataloader("data")


if __name__ == "__main__":
    train()
