import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512
LAYERS = 3
KERNEL_SIZES = [1, 3, 3]
CHANNELS_EXP = 2
LRELU_ALPHA = 0.2
POOLING = torch.nn.MaxPool2d
LR_INIT = 1e-3
LR_DECAY = 0.95
LOSS = torch.nn.L1Loss
ADAM_BETAS = (0.8, 0.999)
