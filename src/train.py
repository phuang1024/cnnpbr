import os

import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


def train():
    dataloader = get_dataloader("train_data")

    model = ColorToDisp().to(device)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        for i, (img, disp) in enumerate(dataloader):
            img = img.to(device)
            disp = disp.to(device)

            pred = model(img)
            loss = loss_fn(pred, disp)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss}")

    return model


if __name__ == "__main__":
    model = train()
    print("Saving model")
    torch.save(model.state_dict(), "model.pth")
    print("Done")
