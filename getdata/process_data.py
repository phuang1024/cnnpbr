import argparse
import os
import time
from threading import Thread

import cv2
import numpy as np
from tqdm import tqdm

import torch


def load_image(args, path):
    name = os.path.splitext(os.path.basename(path))[0]
    grayscale = "disp" in name.lower() or "rough" in name.lower()

    img = cv2.imread(path).astype(float)
    if grayscale:
        img[..., 0] = (img[..., 1] + img[..., 2] + img[..., 0]) / 3
        img = img[..., 0]

    img = img.astype(np.uint8)
    img = cv2.resize(img, (args.size, args.size))
    return img


def prepare_dir(args, dir, new):
    for name in ("color", "normal", "disp", "rough"):
        old_path = os.path.join(dir, name + ".jpg")
        new_path = os.path.join(new, name + ".jpg")
        img = load_image(args, old_path)
        assert cv2.imwrite(new_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Output image size.")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Number of threads to use.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    threads = [None] * args.jobs
    for d in tqdm(os.listdir(args.input), desc="Preparing data"):
        while True:
            time.sleep(0.001)
            for i in range(args.jobs):
                if threads[i] is None or not threads[i].is_alive():
                    break
            else:
                continue
            break

        dir = os.path.join(args.input, d)
        new = os.path.join(args.output, os.path.basename(dir))
        os.makedirs(new, exist_ok=True)

        thread = Thread(target=prepare_dir, args=(args, dir, new))
        thread.start()
        threads[i] = thread


        """
        maps = {}
        for f in os.listdir(dir):
            if f.endswith(".jpg"):
                img = cv2.imread(os.path.join(dir, f))
                height, width = img.shape[:2]
                if width > height:
                    img = img[:, :height, :]
                elif height > width:
                    img = img[:width, :, :]
                img = cv2.resize(img, (args.size, args.size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                color_img = "color" in f.lower() or "normal" in f.lower()
                if not color_img:
                    img = np.sum(img, axis=2) / 3
                    real = np.empty((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                    real[..., 0] = img
                    img = real

                img = img / 255
                img = img.transpose((2, 0, 1))

                name = os.path.splitext(f)[0]
                maps[name] = img

        color = torch.Tensor(maps["color"]).float()
        with open(os.path.join(new, "input.tensor"), "wb") as f:
            torch.save(color, f)

        other = np.empty((5, args.size, args.size), dtype=np.float64)
        other[0:3] = maps["normal"]
        other[3] = maps["disp"]
        other[4] = maps["rough"]

        other = torch.from_numpy(other).float()
        with open(os.path.join(new, "output.tensor"), "wb") as f:
            torch.save(other, f)
        """


if __name__ == "__main__":
    main()
