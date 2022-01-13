import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

# INFO_NAMES = ["name","type"]

# Creds: http://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
def image_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class SampleDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, transform=None, target_transform=None):
        self.img_info = pd.read_csv(annotations_file)
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        if (idx == len(self)):
            raise StopIteration
        img_path = os.path.join(self.dataset_path, self.img_info.at[idx,'name'])
        image = image_loader(img_path)
        label = self.img_info.at[idx,'type']

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label