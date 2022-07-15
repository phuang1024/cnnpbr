from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset
from network import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(args):
    with open(args.logfile, "w") as f:
        f.write(f"Train start: {datetime.now()}\n")

    dataset = TextureDataset(args.data_path, True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.data_workers, prefetch_factor=4)

    model = Classifier(len(dataset.labels)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in (pbar := trange(args.epochs)):
        pbar.set_description("Training", refresh=True)

        avg_loss = 0.0
        for i, (x, y) in enumerate(dataloader):
            msg = f"epoch {epoch + 1}/{args.epochs}, batch {i + 1}/{len(dataloader)}"
            pbar.set_description(msg, refresh=True)

            x = x.to(device)
            y = y.to(device)

            model.train()
            out = model(x)
            loss = loss_fn(out, y)
            avg_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        avg_loss /= len(dataloader)
        with open(args.logfile, "a") as f:
            f.write(f"epoch {epoch + 1}/{args.epochs}, avg_loss: {avg_loss}\n")
