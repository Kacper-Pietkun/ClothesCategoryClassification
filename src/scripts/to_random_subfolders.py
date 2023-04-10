import os
import sys
import random
import shutil


def split_files_to_subfolders(path, split_no):
    """
    Randomly distribute files in a folder given by path, to newly created subfolders.

    :param path: path to a directory containing files
    :param split_no: number of subfolders that will be created
    """
    file_list = os.listdir(path)
    random.shuffle(file_list)
    template_folder_name = "split"
    for i in range(1, split_no + 1):
        os.mkdir(os.path.join(path, f"{template_folder_name}_{i}"))

    for i, file in enumerate(file_list):
        src = os.path.join(path, file)
        dst = os.path.join(path, f"{template_folder_name}_{i % split_no + 1}", file)
        shutil.move(src, dst)


def is_positive_integer(value):
    """
    Check if given value is an integer greater than 0

    :param value: value to be validated
    :return: True if value is a positive integer, otherwise False
    """
    if isinstance(value, int) and value > 0:
        return True
    return False


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    splits_no = int(args[2])

    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid path.")
    if os.path.isfile(path):
        raise ValueError(f"{path} is not a path to a directory.")
    if is_positive_integer(splits_no) is False:
        raise ValueError(f"{splits_no} is not a positive integer.")
    split_files_to_subfolders(path, splits_no)
