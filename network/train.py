import argparse
from datetime import datetime
import os
import time

import matplotlib.pyplot as plt
from tqdm import trange

import torch
from torch.utils.data import DataLoader, random_split

from dataset import TextureDataset
from network import ColorToDisp

TRAIN_DATA_FACTOR = 0.9

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


def train(args, model):
    with open(args.log, "w") as f:
        f.write(f"Started training at {datetime.now()}\n")

    dataset = TextureDataset("../data/data_resized")
    train_count = int(len(dataset) * TRAIN_DATA_FACTOR)
    train_data, test_data = random_split(dataset, [train_count, len(dataset) - train_count])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print(f"Training samples: {len(train_loader.dataset)}, "
          f"test samples: {len(test_loader.dataset)}")

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
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

            msg = f"Epoch: {epoch}, Batch: {i}, Loss: {loss:.4f}"
            with open(args.log, "a") as f:
                f.write(msg + "\n")

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
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--log", default="train.log", help="Path to log file")
    args = parser.parse_args()

    model = ColorToDisp()
    if args.resume:
        print("Loading model")
        model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)

    losses = train(args, model)

    print("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")

    print("Plotting losses")
    plt.plot(losses)
    plt.show()

    print("Done")
