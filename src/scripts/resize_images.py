import os
import argparse
from PIL import Image
from tqdm import tqdm
from PIL import ImageOps


def resize_images(path, new_width, new_height):
    """
    Resize all images in a given directory.

    :param path: path to a directory containing files
    :param new_width: new width of the resized image
    :param new_height: new height of the resized image
    """
    for file_name in tqdm(os.listdir(path)):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(path, file_name)
            image = Image.open(file_path)
            image = ImageOps.exif_transpose(image)
            resized_image = image.resize((new_width, new_height))
            resized_image.save(file_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Model training")
    argparser.add_argument("--directory-path", type=str, required=True)
    argparser.add_argument("--width", type=int, default=227)
    argparser.add_argument("--height", type=int, default=227)
    args = argparser.parse_args()

    if not os.path.exists(args.directory_path):
        raise ValueError(f"{args.directory_path} is not a valid path.")
    if os.path.isfile(args.directory_path):
        raise ValueError(f"{args.directory_path} is not a path to a directory.")
    resize_images(args.directory_path, args.width, args.height)
