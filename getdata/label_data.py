import argparse
import os
import time
from pathlib import Path

import pygame
pygame.init()

LABELS = sorted((
    "wood",
    "woodfloor",
    "asphalt",
    "grassdirt",
    "bricks",
    "fabric",
    "tiles",
    "marble",
    "rock",
    "decal",
))

FONT = pygame.font.SysFont("ubuntu", 16)


def get_dirs(args):
    for d in args.data_dir.iterdir():
        if d.is_dir():
            done = (d / "label.txt").exists()
            if args.relabel or not done:
                yield d


def draw_labels(surface):
    for i, label in enumerate(LABELS):
        text = FONT.render(f"{i}: {label}", True, (0, 0, 0))
        surface.blit(text, (50, 20 * (i+1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()

    dirs = get_dirs(args)
    last_dir = None

    surface = pygame.display.set_mode((640, 480))
    surface.fill((255, 255, 255))
    draw_labels(surface)

    while True:
        time.sleep(0.05)

        events = pygame.event.get()
        choice = None
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.KEYDOWN:
                for i in range(10):
                    key = getattr(pygame, f"K_{i}")
                    numpad = getattr(pygame, f"K_KP{i}")
                    if event.key in (key, numpad):
                        choice = i
                        break

        if choice is not None or last_dir is None:
            if choice is not None:
                with open(last_dir / "label.txt", "w") as f:
                    f.write(LABELS[choice])
                    f.write("\n")
                print("Wrote", last_dir)

            try:
                last_dir = next(dirs)
            except StopIteration:
                print("Done")
                pygame.quit()
                return

            img = pygame.image.load(last_dir / "color.jpg")
            img = pygame.transform.scale(img, (400, 400))
            surface.blit(img, (200, 40))

        pygame.display.update()


if __name__ == "__main__":
    main()
