import argparse
from datetime import datetime
import os
import time

import matplotlib.pyplot as plt
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


def train(args, model):
    with open(args.log, "w") as f:
        f.write(f"Started training at {datetime.now()}\n")

    dataloader = get_dataloader("../data/train_data_resized", batch_size=args.batch_size)
    print(f"Training on {len(dataloader.dataset)} samples")

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    model.train()

    losses = []
    for epoch in trange(args.epochs, desc="Epoch"):
        start = time.time()
        for i, (img, disp) in enumerate(dataloader):
            pred = model(img)
            loss = loss_fn(pred, disp)

            optim.zero_grad()
            loss.backward()
            optim.step()

            msg = f"Epoch: {epoch}, Batch: {i}, Loss: {loss:.4f}"
            with open(args.log, "a") as f:
                f.write(msg + "\n")

        losses.append(loss)

    plt.plot(losses)
    plt.show()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
        help="Continue training from a previous model")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
    parser.add_argument("--log", default="train.log", help="Path to log file")
    args = parser.parse_args()

    model = ColorToDisp()
    if args.resume:
        print("Loading model")
        model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)

    train(args, model)

    print("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")

    print("Done")
