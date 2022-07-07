import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 1024
LAYERS = 4
