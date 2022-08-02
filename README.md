# CNNPBR

Physically based rendering texture maps using CNNs.

## About

For 3D creation, sometimes textures (images) are used to create a photorealistic look.
Aside from the color information, roughness, displacement (height differences), normal
(surface direction), and ambient occulsion (small color brightness changes) are needed.
These *maps* are also images, but instead of color, they represent their respective
qualities.

This neural network generates the displacement, roughness, or ambient occulsion maps from
the color map. The normal map can be generated from the displacement with another
algorithm (not provided).

## Example

![](https://github.com/phuang1024/cnnpbr/blob/master/results.jpg?raw=true)
