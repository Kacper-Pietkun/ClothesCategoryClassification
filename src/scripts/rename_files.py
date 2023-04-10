import os
import sys
import datetime


def rename_files(path):
    """
    Renames all files in a given directory.

    :param path: path to a directory containing files
    """
    formatted_date = datetime.date.today().strftime("%d_%m_%Y")
    template_name = f"own_img_{formatted_date}"

    counter = 1
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            new_file_name = f"{template_name}_{counter}.jpg"
            new_file_path = os.path.join(path, new_file_name)
            try:
                os.rename(file_path, new_file_path)
            except FileExistsError as e:
                ...
            counter += 1


if __name__ == "__main__":
    args = sys.argv
    path = args[1]

    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid path.")
    if os.path.isfile(path):
        raise ValueError(f"{path} is not a path to a directory.")
    rename_files(path)

