import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset, extract_maps
from network import CNNPBRModel

NUM_PER_SHOW = 4


def test(model):
    dataset = TextureDataset("../data/train_resized")
    dataloader = DataLoader(dataset, batch_size=NUM_PER_SHOW)

    # Evaluate each image in the dataset and plot test, ground truth, and prediction
    next_i = 0
    for color, truth in dataloader:
        pred = model(color)
        color, pred, truth = color.detach().cpu().numpy(), pred.detach().cpu().numpy(), \
                truth.detach().cpu().numpy()
        color = color.transpose((0, 2, 3, 1))
        pred = pred.transpose((0, 2, 3, 1))
        truth = truth.transpose((0, 2, 3, 1))

        # Display images
        for i in range(color.shape[0]):
            curr_pred = extract_maps(pred[i])
            curr_truth = extract_maps(truth[i])

            plt.subplot(4, NUM_PER_SHOW*2, 2*i+1)
            plt.imshow(color[i])
            plt.title("Input (color)")
            plt.axis("off")

            for j, name in enumerate(curr_pred):
                ind = (j+1)*NUM_PER_SHOW*2 + 2*i + 1

                plt.subplot(4, NUM_PER_SHOW*2, ind)
                plt.imshow(curr_truth[name].astype(np.uint8))
                plt.title(name + "(truth)")
                plt.axis("off")

                plt.subplot(4, NUM_PER_SHOW*2, ind+1)
                plt.imshow(curr_pred[name].astype(np.uint8))
                plt.title(name + "(pred)")
                plt.axis("off")

    plt.show()


if __name__ == "__main__":
    model = CNNPBRModel(layers=LAYERS).to(DEVICE)
    model.load_state_dict(torch.load("model.pth"))
    test(model)
