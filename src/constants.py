# Global params
IMG_SIZE = 256

# Data augmentation
AUG_JITTER = (0.08, 0.08, 0.08, 0.05)
AUG_SHARP = 2
AUG_NOISE = 0.03

# Network params
NET_LAYERS = 8
NET_BN_MOMENTUM = 0.1
NET_CONV_CH = 32
NET_CONV_LAYERS = 3     # Each NxConv layer count.
NET_PRECONV_LAYERS = 2
NET_CONV_KERNEL = 3
NET_ELU_ALPHA = 1
