import argparse
import os

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-s", "--size", type=int, default=256, help="Output image size.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for d in tqdm(os.listdir(args.input), desc="Preparing data"):
        dir = os.path.join(args.input, d)
        new = os.path.join(args.output, os.path.basename(dir))
        os.makedirs(new, exist_ok=True)
        for f in os.listdir(dir):
            if f.endswith(".jpg"):
                read_mode = cv2.IMREAD_COLOR if "color" in f.lower() or "normal" in f.lower() \
                    else cv2.IMREAD_GRAYSCALE
                img = cv2.imread(os.path.join(dir, f), read_mode)
                img = cv2.resize(img, (args.size, args.size))
                cv2.imwrite(os.path.join(new, f), img)


if __name__ == "__main__":
    main()
