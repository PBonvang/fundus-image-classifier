import config
from torch.utils.data import DataLoader, Subset
from utils.FundusDataset import FundusDataset
from torchvision import datasets
import os

def get_dataset(info_file, data_dir, transforms):
	ds = FundusDataset(info_file, data_dir, transforms)

	return ds