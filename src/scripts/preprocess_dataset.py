import os
import sys
import numpy as np
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from src.classes.clothes_dataset import ClothesDataset


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1])
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid path.")
    if os.path.isfile(path):
        raise ValueError(f"{path} is not a path to a directory.")

    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ClothesDataset(path, transform=my_transforms)
    print("Calculating mean")
    mean = np.mean([np.mean(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)], axis=0)
    print("Calculating std")
    std = np.mean([np.std(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)], axis=0)

    updated_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = ClothesDataset(path, transform=updated_transforms)
    print("Calculating mean")
    mean = np.mean([np.mean(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)], axis=0)
    print("Calculating std")
    std = np.mean([np.std(x.numpy(), axis=(1, 2)) for x, _ in tqdm(dataset)], axis=0)



