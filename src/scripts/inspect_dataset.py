import os
import sys
import pathlib
import matplotlib.pyplot as plt


def retrieve_info(path):
    """
    Get information about a dataset specified by given path

    :param path: path to the dataset's root folder
    :return:
        - dataset_name - Name of the dataset
        - class_names - List of classes from a dataset sorted by name
        - count_by_class - List of images for each class sorted according to class_names
    """
    dataset_name = str(path).split("\\")[-1]
    class_names = sorted([item.name for item in path.glob("*")])
    count_by_class = [len(item) for item in (list(path.glob(f"{class_name}/*.jpg")) for class_name in class_names)]
    return dataset_name, class_names, count_by_class


def text_summary(dataset_name, class_names, count_by_class):
    """
    Print information about the dataset to the console

    :param dataset_name: Name of the dataset
    :param class_names: List of classes from a dataset sorted by name
    :param count_by_class: List of images for each class sorted according to class_names
    """
    print("---------------------------")
    print(f"Dataset - {dataset_name}")
    print("---------------------------")
    for name, count in zip(class_names, count_by_class):
        print(f"{name}: {count}")
    print("---------------------------")
    print(f"Sum: {sum(count_by_class)}")
    print("---------------------------")


def plot_statistics(dataset_name, class_names, count_by_class):
    """
    Bar plot of dataset's statistics

    :param dataset_name: Name of the dataset
    :param class_names: List of classes from a dataset sorted by name
    :param count_by_class: List of images for each class sorted according to class_names
    """
    fig, ax = plt.subplots()
    ax.bar(class_names, count_by_class)
    ax.set_title(f"Statistics of {dataset_name}")
    ax.set_xlabel("Class name")
    ax.set_ylabel("Image count")
    plt.show()


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1])
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid path.")
    if os.path.isfile(path):
        raise ValueError(f"{path} is not a path to a directory.")
    dataset_name, class_names, count_by_class = retrieve_info(path)
    text_summary(dataset_name, class_names, count_by_class)
    plot_statistics(dataset_name, class_names, count_by_class)
