# Global params
IMG_SIZE = 256

# Data augmentation
AUG_JITTER = (0.08, 0.08, 0.08, 0.05)
AUG_SHARP = 2
AUG_NOISE = 0.03

# Network params
NET_LAYERS = 6
NET_CONV_CH = 4         # Initial number of channels.
NET_CONV_LAYERS = 3     # Each NxConv layer count.
NET_PRECONV_CH = 8
NET_PRECONV_LAYERS = 4
