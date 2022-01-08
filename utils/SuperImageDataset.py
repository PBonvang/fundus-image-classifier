import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

# INFO_NAMES = ["index","img_dir","img_labels"]


class SuperImageDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, transform=None, target_transform=None):
        self.img_info = pd.read_csv(annotations_file)
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_info.at[idx,'img_dir'])
        image = read_image(img_path)
        label = self.img_info.at[idx,'img_labels']

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label