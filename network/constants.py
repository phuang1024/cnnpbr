import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512
LAYERS = 3
KERNEL_SIZES = [3, 5, 5]
CHANNELS_EXP = 2
LU_ALPHA = 0.2
POOLING = torch.nn.MaxPool2d
LR_INIT = 1e-3
LR_DECAY = 0.95
LOSS = torch.nn.MSELoss
ADAM_BETAS = (0.8, 0.999)
