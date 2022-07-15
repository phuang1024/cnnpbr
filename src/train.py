import matplotlib.pyplot as plt
import numpy as np

from dataset import TextureDataset


def train_model(args):
    dataset = TextureDataset(args.data_path, True)
    for i in range(8):
        img, label = dataset[0]
        img = img.detach().numpy()
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)

        plt.subplot(4, 2, i+1)
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(label)

    plt.show()
