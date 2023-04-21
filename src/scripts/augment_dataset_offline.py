import os
import sys
import pathlib
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.classes.clothes_dataset import ClothesDataset
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    path_dataset = pathlib.Path(sys.argv[1])
    path_augmented_dataset = pathlib.Path(sys.argv[2])
    if not os.path.exists(path_dataset):
        raise ValueError(f"{path_dataset} is not a valid path.")
    if os.path.isfile(path_dataset):
        raise ValueError(f"{path_dataset} is not a path to a directory.")
    if not os.path.exists(path_augmented_dataset):
        raise ValueError(f"{path_augmented_dataset} is not a valid path.")

    standard_transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.ToTensor()
    ])
    dataset = ClothesDataset(path_dataset, transform=standard_transform)
    class_counts = dataset.count_by_class
    print("Calculating samples' weights")
    class_weights = [1.0 / class_counts[np.argmax(label)] for _, label in tqdm(dataset)]

    try:
        for category in dataset.classes:
            category_path = os.path.join(path_augmented_dataset, category)
            pathlib.Path(category_path).mkdir(exist_ok=False)
    except FileExistsError as e:
        print(f"Augmented dataset may already exists, make sure you will not delete existing images: {category_path}, {e}")
        sys.exit(-1)

    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.CenterCrop(224)
    ])

    sampler = WeightedRandomSampler(class_weights, len(dataset), replacement=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, shuffle=False)

    formatted_date = datetime.date.today().strftime("%d_%m_%Y")
    template_name = f"augmented_img_{formatted_date}"
    for i, (image, label) in enumerate(tqdm(dataloader)):
        class_name = dataset.reverse_mapping[np.argmax(label).item()]
        transformed_image = augmentation_transform(image)
        pil_image = transforms.ToPILImage()(transformed_image.squeeze())
        pil_image.save(f"{os.path.join(path_augmented_dataset, class_name, template_name)}_{i+1}.jpg")
