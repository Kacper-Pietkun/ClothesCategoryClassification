import sys
import os
import json
import pathlib
import shutil
from tqdm import tqdm


def determine_set(path_dataset, image_name):
    train_path = os.path.join(path_dataset_original, "Train", f"{key}.jpg")
    val_path = os.path.join(path_dataset_original, "Validation", f"{key}.jpg")

    if os.path.isfile(train_path):
        return "Train"
    if os.path.isfile(val_path):
        return "Validation"
    return "Test"


if __name__ == "__main__":

    path_dataset_original = pathlib.Path(sys.argv[1])
    path_dataset_to_create = pathlib.Path(sys.argv[2])
    json_metadata_path = pathlib.Path(sys.argv[3])
    if not os.path.exists(path_dataset_to_create):
        raise ValueError(f"{path_dataset_to_create} is not a valid path.")
    if not os.path.isfile(json_metadata_path):
        raise ValueError(f"{json_metadata_path} is not a file.")

    with open(json_metadata_path, "r") as f:
        data = json.load(f)
        for key in tqdm(data.keys()):
            category = data[key]["category"]
            image_name = f"{key}.jpg"
            set_type = determine_set(path_dataset_original, image_name)
            file_path = os.path.join(path_dataset_original, set_type, image_name)
            set_type_path = os.path.join(path_dataset_to_create, set_type)
            pathlib.Path(set_type_path).mkdir(exist_ok=True)
            final_path = os.path.join(path_dataset_to_create, set_type, category)
            pathlib.Path(final_path).mkdir(exist_ok=True)
            shutil.copy(file_path, final_path)
