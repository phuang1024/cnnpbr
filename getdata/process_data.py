import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Output image size.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for d in tqdm(os.listdir(args.input), desc="Preparing data"):
        dir = os.path.join(args.input, d)
        new = os.path.join(args.output, os.path.basename(dir))
        os.makedirs(new, exist_ok=True)
        for f in os.listdir(dir):
            if f.endswith(".jpg"):
                img = cv2.imread(os.path.join(dir, f))
                height, width = img.shape[:2]
                if width > height:
                    img = img[:, :height, :]
                elif height > width:
                    img = img[:width, :, :]

                color_img = "color" in f.lower() or "normal" in f.lower()
                if not color_img:
                    img = np.sum(img, axis=2) / 3

                img = cv2.resize(img, (args.size, args.size))
                cv2.imwrite(os.path.join(new, f), img)


if __name__ == "__main__":
    main()
