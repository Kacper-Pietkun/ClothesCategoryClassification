import os
import sys
import pathlib
import matplotlib.pyplot as plt
from src.classes.clothes_dataset import ClothesDataset


def text_summary(dataset):
    """
    Print information about the dataset to the console

    :param dataset: dataset instance
    """
    print("---------------------------")
    print(f"Dataset - {dataset.name}")
    print("---------------------------")
    for name, count in zip(dataset.classes, dataset.count_by_class):
        print(f"{name}: {count}")
    print("---------------------------")
    print(f"Sum: {len(dataset)}")
    print("---------------------------")


def plot_statistics(dataset):
    """
    Bar plot of dataset's statistics

    :param dataset: dataset instance
    """
    fig, ax = plt.subplots()
    ax.bar(dataset.classes, dataset.count_by_class)
    ax.set_title(f"Statistics of {dataset.name}")
    ax.set_xlabel("Class name")
    ax.set_ylabel("Image count")
    plt.show()


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1])
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid path.")
    if os.path.isfile(path):
        raise ValueError(f"{path} is not a path to a directory.")
    dataset = ClothesDataset(path)
    text_summary(dataset)
    plot_statistics(dataset)
