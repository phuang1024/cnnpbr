import argparse
from pathlib import Path

from getdata import getdata


def get_args():
    parser = argparse.ArgumentParser(description="PBR texture generation using CNN.")
    subp = parser.add_subparsers(dest="action")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the data directory.")
    parser.add_argument("--results-path", type=Path, required=True, help="Directory to save results.")
    parser.add_argument("--tmp-path", type=Path, default="/tmp", help="Temp directory.")

    datap = subp.add_parser("data", help="Data download from ambientcg.")
    datap.add_argument("-c", "--count", type=int, default=10, help="Number of textures to download.")
    datap.add_argument("-s", "--size", type=int, default=1024, help="Size of the textures.")
    datap.add_argument("--split", type=float, default=0.9, help="Train/test split.")
    datap.add_argument("--category", type=str, required=True, help="Category and label.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.action is None:
        print("No action specified.")
        return

    elif args.action == "data":
        getdata(args)


if __name__ == "__main__":
    main()
