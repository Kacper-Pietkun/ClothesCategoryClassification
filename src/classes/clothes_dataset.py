import os
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
import torch


class ClothesDataset(Dataset):
    def __init__(self, root_folder, transform=None, target_transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.target_transform = target_transform
        self.name = str(root_folder).split("\\")[-1]
        self.classes = sorted([item.name for item in root_folder.glob("*")])
        self.class_mapping = {self.classes[i]: i for i in range(len(self.classes))}
        self.reverse_mapping = {i: self.classes[i] for i in range(len(self.classes))}
        self.count_by_class = [len(item) for item in (list(root_folder.glob(f"{category}/*.jpg")) for
                                                      category in self.classes)]
        self.image_rel_paths = []
        self.labels = []
        for index, category in enumerate(self.classes):
            for file_name in os.listdir(os.path.join(root_folder, category)):
                self.image_rel_paths.append(os.path.join(category, file_name))
                self.labels.append(index)
        self.one_hot_labels = torch.eye(len(self.classes))[self.labels]

    def __len__(self):
        return len(self.image_rel_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_folder, self.image_rel_paths[idx])
        label = self.one_hot_labels[idx]
        image = Image.open(str(image_path))
        image = ImageOps.exif_transpose(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, label
