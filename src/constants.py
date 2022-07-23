# Global params
IMG_SIZE = 256

# Data augmentation
AUG_JITTER = (0.1, 0.1, 0.1, 0.05)
AUG_SHARP = 2
AUG_NOISE = 0.1

# Network params
NET_LAYERS = 7
NET_BN_MOMENTUM = 0.5
NET_CONV_CH = 16
NET_CONV_LAYERS = 3     # Each NxConv layer count.
NET_CONV_KERNEL = 3
NET_LRELU_ALPHA = 0.1
