import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import DataLoader

import pytorch_ssim
from constants import *
from dataset import TextureDataset
from network import Network

ROOT = Path(__file__).absolute().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


class LossFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.ssim = pytorch_ssim.SSIM()

    def forward(self, x, y):
        mse = self.mse(x, y)
        ssim = self.ssim(x, y)
        return mse + 0.1/ssim


def get_session_path(args):
    sess = []
    for d in args.results_path.iterdir():
        sess.append(int(d.name))
    next_sess = max(sess) + 1 if sess else 1

    return args.results_path / f"{next_sess:03d}"


def save_image(img, path):
    """
    Image shape: (3, H, W)
    """
    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255.0
    img = img.astype(np.uint8)
    plt.imsave(str(path), img)


def train_model(args):
    session_path = get_session_path(args)
    session_path.mkdir(parents=True, exist_ok=True)
    log_path = session_path / "train.log"

    dataset = TextureDataset(args.data_path, False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.data_workers, prefetch_factor=4)

    model = Network().to(device)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    # Write info to results folder
    print("Saving results to:", session_path)
    with log_path.open("w") as f:
        f.write(f"Train start: {datetime.now()}\n")
        f.write(f"Train samples: {len(dataset)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n\n")
        f.write(f"Train progress:\n")
    with (session_path/"model.txt").open("w") as f:
        f.write(str(model))
    with (session_path/"commit.txt").open("w") as f:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        f.write(commit + "\n")
    shutil.copyfile(ROOT/"constants.py", session_path/"constants.py")

    for epoch in (pbar := trange(args.epochs, desc="Training")):
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
        losses.append(avg_loss)
        with log_path.open("a") as f:
            f.write(f"epoch {epoch + 1}/{args.epochs}, avg_loss: {avg_loss}\n")
        save_path = session_path / "models" / f"epoch_{epoch + 1}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    plt.plot(losses)
    plt.savefig(str(session_path / "loss.jpg"))
