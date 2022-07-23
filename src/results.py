import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset
from network import Network


def get_model_path(session, epoch, results_path):
    if session == -1:
        session = max(map(int, (p.name for p in results_path.iterdir())))
    models_path = results_path / f"{session:03d}" / "models"

    if epoch == -1:
        epoch = max(int(p.stem.split("_")[-1]) for p in models_path.iterdir())
    return models_path / f"epoch_{epoch}.pt"


def show_results(args):
    model = Network()
    params_path = get_model_path(args.session, args.epoch, args.results_path)
    print(f"Loading model from {params_path}")
    model.load_state_dict(torch.load(params_path))

    plt.axis("off")

    dataset = TextureDataset(args.data_path, False)

    next_i = 0
    for color, truth in dataset:
        color = color.unsqueeze(0)
        truth = truth.unsqueeze(0)

        pred = model(color)
        color, pred, truth = color.detach().numpy(), pred.detach().numpy(), truth.numpy()

        color = ((color * 0.5 + 0.5) * 255).astype(np.uint8)
        pred = ((pred * 0.5 + 0.5) * 255).astype(np.uint8)
        truth = ((truth * 0.5 + 0.5) * 255).astype(np.uint8)

        color = color.transpose((0, 2, 3, 1))
        pred = pred.transpose((0, 2, 3, 1))
        truth = truth.transpose((0, 2, 3, 1))
        loss = np.power(pred - truth, 2)

        plt.subplot(4, 8, next_i + 1)
        plt.imshow(color[0])
        plt.title("Input (color)")

        plt.subplot(4, 8, 8 + next_i + 1)
        plt.imshow(pred[0])
        plt.title("Prediction (disp)")

        plt.subplot(4, 8, 16 + next_i + 1)
        plt.imshow(truth[0])
        plt.title("Truth (disp)")

        plt.subplot(4, 8, 24 + next_i + 1)
        plt.imshow(loss[0])
        plt.title("Loss (disp)")

        next_i += 1

        if next_i == 8:
            next_i = 0
            plt.show()
