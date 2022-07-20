# Global params
IMG_SIZE = 256

# Data augmentation
AUG_JITTER = (0.1, 0, 0.1, 0.05)
AUG_SHARP = 1

# Network params
NET_LAYERS = 6
NET_CONV_CH = 16
NET_CONV_LAYERS = 3     # Each NxConv layer count.
NET_CONV_KERNEL = 3
NET_LRELU_ALPHA = 0.1
