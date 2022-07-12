import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512
LAYERS = 3
KERNEL_SIZE = 3
CHANNELS_EXP = 2
LRELU_ALPHA = 0.2
LR_INIT = 2e-4
LR_DECAY = 0.8
LOSS = torch.nn.L1Loss
OPTIM_BETAS = (0.5, 0.999)
