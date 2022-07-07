import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import TextureDataset
from network import CNNPBRModel


def test(model):
    dataset = TextureDataset("../data/test_resized")
    dataloader = DataLoader(dataset, batch_size=10)

    # Evaluate each image in the dataset and plot test, ground truth, and prediction
    count = 10
    next_i = 0
    for color, disp in dataloader:
        color, disp = color.to(DEVICE), disp.to(DEVICE)
        pred = model(color) * 255
        color, disp, pred = \
            color.detach().cpu().numpy(), disp.detach().cpu().numpy(), pred.detach().cpu().numpy()

        # Display images
        for i in range(color.shape[0]):
            plt.subplot(3, count, next_i+1)
            plt.imshow(color[i].transpose(1, 2, 0))
            plt.subplot(3, count, count+next_i+1)
            plt.imshow(disp[i].transpose(1, 2, 0))
            plt.subplot(3, count, 2*count+next_i+1)
            plt.imshow(pred[i].transpose(1, 2, 0))
            next_i += 1
            if next_i == count:
                plt.show()
                next_i = 0

    plt.show()


if __name__ == "__main__":
    model = CNNPBRModel(layers=LAYERS)
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(DEVICE)
    test(model)
