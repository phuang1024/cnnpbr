import matplotlib.pyplot as plt

import torch

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(model):
    dataloader = get_dataloader("../data/test_data_resized", batch_size=10)

    # Evaluate each image in the dataset and plot test, ground truth, and prediction
    count = len(dataloader.dataset)
    next_i = 0
    for color, disp in dataloader:
        color, disp = color.to(device), disp.to(device)
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

    plt.show()


if __name__ == "__main__":
    model = ColorToDisp()
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)
    test(model)
