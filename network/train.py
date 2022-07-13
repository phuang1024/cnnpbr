import argparse
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset
from network import CNNPBRModel


def train(args, model):
    torch.multiprocessing.set_start_method("spawn")

    print("Batch size:", args.batch_size)
    with open(args.log, "w") as f:
        f.write(f"Started training at {datetime.now()}\n")

    train_data = TextureDataset("../data/train_data")
    test_data = TextureDataset("../data/test_data")
    kwargs = {"shuffle": True, "batch_size": args.batch_size, "num_workers": args.data_workers}
    train_loader = DataLoader(train_data, **kwargs)
    test_loader = DataLoader(test_data, **kwargs)
    print(f"Training samples: {len(train_loader.dataset)}, "
          f"test samples: {len(test_loader.dataset)}")

    loss_fn = LOSS()
    lr = LR_INIT * LR_DECAY ** args.start_epoch
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=ADAM_BETAS)
    scheduler = ExponentialLR(optim, gamma=LR_DECAY)

    print(f"Using loss function {loss_fn}")
    print(f"Using optimizer {optim}")
    print(model)

    losses = []
    for epoch in range(args.epochs):
        # Train model
        model.train()
        for i, (in_data, truth) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            """
            img = in_data[0].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img*255).astype(np.uint8)
            cv2.imwrite("a.png", img)
            stop
            """
            in_data, truth = in_data.to(DEVICE), truth.to(DEVICE)

            pred = model(in_data)
            loss = loss_fn(pred, truth)

            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        # Compute average loss on test dataset
        avg_loss = 0
        model.eval()
        for in_data, truth in test_loader:
            in_data, truth = in_data.to(DEVICE), truth.to(DEVICE)

            pred = model(in_data)
            avg_loss += loss_fn(pred, truth).item()
        avg_loss /= len(test_loader)

        msg = f"Epoch: {epoch}, Loss: {avg_loss:.4f}, LR: {optim.param_groups[0]['lr']:.2e}"
        with open(args.log, "a") as f:
            f.write(msg + "\n")
        losses.append(avg_loss)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Continue training from a previous model")
    parser.add_argument("--start-epoch", type=int, default=0, help="If resuming, how many epochs already done?")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--data-workers", default=0, type=int, help="Number of data workers")
    parser.add_argument("--log", default="train.log", help="Path to log file")
    args = parser.parse_args()

    print(f"Using device {DEVICE}")

    model = CNNPBRModel().to(DEVICE)
    if args.resume:
        print("Loading model")
        model.load_state_dict(torch.load("model.pth"))

    losses = train(args, model)

    print("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")

    print("Plotting losses")
    plt.plot(losses)
    plt.show()

    print("Done")
