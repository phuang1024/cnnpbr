import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset
from network import Network
from results import get_model_path

ROOT = Path(__file__).absolute().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    # Create session directory
    session_path = get_session_path(args)
    session_path.mkdir(parents=True, exist_ok=True)
    log_path = session_path / "train.log"

    # Get data
    dataset = TextureDataset(args.data_path)
    train_size = int(len(dataset) * args.train_split)
    test_size = len(dataset) - train_size
    loader_args = {"batch_size": args.batch_size, "shuffle": True, "pin_memory": True,
            "prefetch_factor": 4, "num_workers": args.data_workers}

    # Create network
    model = Network().to(device)
    loss_fn = torch.nn.HuberLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []

    # Write info to results folder
    if args.resume != -1:
        last_path = get_model_path(args.resume, -1, args.results_path)
        print(f"Loading model from {last_path}")
        model.load_state_dict(torch.load(last_path))

    print("Saving results to:", session_path)
    with log_path.open("w") as f:
        f.write(f"Train start: {datetime.now()}\n")
        f.write(f"Train samples: {train_size}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr:2.2e}\n")
        f.write(f"Weight decay: {args.weight_decay:2.2e}\n")
        f.write(f"Epochs: {args.epochs}\n")
        if args.resume == -1:
            f.write("No resume\n\n")
        else:
            f.write(f"Resume from session {args.resume}\n\n")
        f.write(f"Train progress:\n")
    with (session_path/"model.txt").open("w") as f:
        f.write(str(model))
    with (session_path/"commit.txt").open("w") as f:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        f.write(commit + "\n")
    shutil.copyfile(ROOT/"constants.py", session_path/"constants.py")

    for epoch in (pbar := trange(args.epochs, desc="Training")):
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data.augment = True
        test_data.augment = False
        train_loader = DataLoader(train_data, **loader_args)
        test_loader = DataLoader(test_data, **loader_args)

        # Train
        model.train()
        model.zero_grad()
        optim.zero_grad()
        train_loss = []
        for i, (x, y) in enumerate(train_loader):
            msg = f"Train: epoch {epoch + 1}/{args.epochs}, batch {i + 1}/{len(train_loader)}"
            pbar.set_description(msg, refresh=True)

            x = x.to(device)
            y = y.to(device)
            out = model(x)

            loss = loss_fn(out, y)
            train_loss.append(loss.item())
            loss /= args.batch_step  # Normalize loss by batches per step
            loss.backward()

            if (i+1) % args.batch_step == 0:
                optim.step()
                optim.zero_grad()
                model.zero_grad()
        train_loss = sum(train_loss) / len(train_loss)

        # Evaluate
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for i, (x, y) in enumerate(test_loader):
                msg = f"Eval: epoch {epoch + 1}/{args.epochs}, batch {i + 1}/{len(test_loader)}"
                pbar.set_description(msg, refresh=True)
    
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                test_loss += loss_fn(out, y).item() / len(test_loader)

        # Save progress
        losses.append((train_loss, test_loss))
        with log_path.open("a") as f:
            f.write(f"epoch {epoch + 1}/{args.epochs}, train_loss: {train_loss}, test_loss: {test_loss}\n")
        if epoch % args.save_every == 0:
            save_path = session_path / "models" / f"epoch_{epoch + 1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    # Plot losses
    losses = np.array(losses)
    plt.subplot(211)
    plt.plot(losses[:, 0], label="train")

    plt.subplot(212)
    plt.plot(losses[:, 1], label="test")

    plt.savefig(str(session_path / "loss.jpg"))
