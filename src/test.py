import matplotlib.pyplot as plt

import torch

from dataset import get_dataloader
from network import ColorToDisp

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(model):
    dataloader = get_dataloader("test_data", batch_size=64)

    # Evaluate each image in the dataset and plot test, ground truth, and prediction
    for color, disp in dataloader:
        color, disp = color.to(device), disp.to(device)
        pred = model(color) * 255
        color, disp, pred = \
            color.detach().cpu().numpy(), disp.detach().cpu().numpy(), pred.detach().cpu().numpy()

        # Display images
        count = min(6, color.shape[0])
        for i in range(count):
            plt.subplot(3, count, i+1)
            plt.imshow(color[i].transpose(1, 2, 0))
            plt.subplot(3, count, count+i+1)
            plt.imshow(disp[i].transpose(1, 2, 0))
            plt.subplot(3, count, 2*count+i+1)
            plt.imshow(pred[i].transpose(1, 2, 0))

        plt.show()


if __name__ == "__main__":
    model = ColorToDisp()
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)
    test(model)
