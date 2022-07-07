import argparse
from datetime import datetime
import os
import time

import matplotlib.pyplot as plt
from tqdm import trange

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

from constants import *
from dataset import TextureDataset
from network import CNNPBRModel


def train(args, model):
    print("Batch size:", args.batch_size)
    with open(args.log, "w") as f:
        f.write(f"Started training at {datetime.now()}\n")

    train_data = TextureDataset("../data/train_resized")
    test_data = TextureDataset("../data/test_resized")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print(f"Training samples: {len(train_loader.dataset)}, "
          f"test samples: {len(test_loader.dataset)}")

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optim, gamma=0.997)
    print(f"Using loss function {loss_fn}")
    print(f"Using optimizer {optim}")
    print(model)

    losses = []
    for epoch in trange(args.epochs, desc="Epoch"):
        # Train model
        model.train()
        for i, (img, disp) in enumerate(train_loader):
            pred = model(img)
            loss = loss_fn(pred, disp)

            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        msg = f"Epoch: {epoch}, Loss: {loss:.4f}, LR: {optim.param_groups[0]['lr']:.6f}"
        with open(args.log, "a") as f:
            f.write(msg + "\n")

        if epoch % 100 == 0:
            path = os.path.join("/home/patrick/stuff/cnnpbr", f"model_{epoch}.pth")
            torch.save(model.state_dict(), path)

        # Compute average loss on test dataset
        avg_loss = 0
        model.eval()
        for img, disp in test_loader:
            pred = model(img)
            avg_loss += loss_fn(pred, disp).item()
        avg_loss /= len(test_loader)
        losses.append(avg_loss)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
        help="Continue training from a previous model")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--log", default="train.log", help="Path to log file")
    args = parser.parse_args()

    print(f"Using device {DEVICE}")

    model = CNNPBRModel(layers=LAYERS)
    if args.resume:
        print("Loading model")
        model.load_state_dict(torch.load("model.pth"))
    model = model.to(DEVICE)

    losses = train(args, model)

    print("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")

    print("Plotting losses")
    plt.plot(losses)
    plt.show()

    print("Done")
