import argparse
from pathlib import Path

from getdata import getdata
from results import show_results
from train import train_model


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

    trainp = subp.add_parser("train", help="Train a model.")
    trainp.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    trainp.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    trainp.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    trainp.add_argument("--data-workers", type=int, default=4, help="Number of data workers.")
    trainp.add_argument("--resume", type=int, default=-1, help="Resume from this session, last epoch.")

    resultsp = subp.add_parser("results", help="Generate results.")
    resultsp.add_argument("--session", type=int, default=-1, help="Session number.")
    resultsp.add_argument("--epoch", type=int, default=-1, help="Epoch number.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.action is None:
        print("No action specified.")
        return

    elif args.action == "data":
        getdata(args)

    elif args.action == "train":
        train_model(args)

    elif args.action == "results":
        show_results(args)


if __name__ == "__main__":
    main()
