import os
import sys
import json
import shutil
import pathlib
import argparse
from tqdm import tqdm


def determine_set(path_dataset, image_name):
    """
    Looking for a given image in a original dataset, to determine if it is in train, validation or test set

    :param path_dataset: path to the original dataset
    :param image_name: name of the image to look for
    :return: String representing "Train", "Validation" or "Test" set
    """
    train_path = os.path.join(path_dataset, "Train", image_name)
    if os.path.isfile(train_path):
        return "Train"
    val_path = os.path.join(path_dataset, "Validation", image_name)
    if os.path.isfile(val_path):
        return "Validation"
    return "Test"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Model training")
    argparser.add_argument("--dataset-original-path", type=str, required=True)
    argparser.add_argument("--dataset-new-path", type=str, required=True)
    argparser.add_argument("--metadata-path", type=str, required=True)
    args = argparser.parse_args()

    if not os.path.exists(args.dataset_original_path):
        raise ValueError(f"{args.dataset_original_path} is not a valid path.")
    if os.path.isfile(args.dataset_original_path):
        raise ValueError(f"{args.dataset_original_path} is not a path to a directory.")
    if not os.path.exists(args.dataset_new_path):
        raise ValueError(f"{args.dataset_new_path} is not a valid path.")
    if not os.path.isfile(args.metadata_path):
        raise ValueError(f"{args.metadata_path} is not a file.")

    with open(args.metadata_path, "r") as f:
        data = json.load(f)
        for key in tqdm(data.keys()):
            category = data[key]["category"]
            image_name = f"{key}.jpg"
            set_type = determine_set(args.dataset_original_path, image_name)
            file_path = os.path.join(args.dataset_original_path, set_type, image_name)
            set_type_path = os.path.join(args.dataset_new_path, set_type)
            pathlib.Path(set_type_path).mkdir(exist_ok=True)
            final_path = os.path.join(args.dataset_new_path, set_type, category)
            pathlib.Path(final_path).mkdir(exist_ok=True)
            shutil.copy(file_path, final_path)
