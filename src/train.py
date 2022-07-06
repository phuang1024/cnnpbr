import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


def train(model):
    dataloader = get_dataloader("../data/train_data_resized", batch_size=128)
    print(f"Training on {len(dataloader.dataset)} samples")

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(model)

    model.train()
    for epoch in range(200):
        start = time.time()
        for i, (img, disp) in enumerate(dataloader):
            pred = model(img)
            loss = loss_fn(pred, disp)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(f"\rEpoch: {epoch}, Batch: {i}, Loss: {loss:.4f}", end="", flush=True)

        print(f" Time: {time.time() - start:.2f}")

    return model


if __name__ == "__main__":
    print("Loading model")
    model = ColorToDisp()
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)

    train(model)

    print("Saving model")
    torch.save(model.state_dict(), "model.pth")

    print("Done")
