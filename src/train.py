from dataset import TextureDataset


def train_model(args):
    dataset = TextureDataset(args.data_path, True)
    print(dataset[0])
