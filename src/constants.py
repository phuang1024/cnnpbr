# Global params
IMG_SIZE = 256

# Data augmentation
AUG_JITTER = (0.1, 0.1, 0.1, 0.05)
AUG_SHARP = 2

# Network params
NET_LAYERS = 7
NET_CONV_CH = 32
NET_CONV_LAYERS = 4     # Each NxConv layer count.
NET_CONV_KERNEL = 3
NET_LRELU_ALPHA = 0.1
