import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DATA_SPLIT = 0.9

IMG_SIZE = 256
LAYERS = 5
