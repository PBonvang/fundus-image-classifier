import config
from torch.utils.data import DataLoader, Subset
from utils.SuperImageDataset import SuperImageDataset
from torchvision import datasets
import os

def get_super_dataloader(info_file, data_dir, transforms, batch_size, shuffle=True, limit=0):
    ds = SuperImageDataset(info_file, data_dir, transforms)
    if limit:
        ds = Subset(ds, list(range(limit)))
        
    loader = DataLoader(ds, batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if config.DEVICE == "cuda" else False)

    return (ds, loader)

def get_sample_dataloader(data_dir, transforms, batch_size, shuffle=True, limit=0):
	ds = datasets.ImageFolder(root=data_dir,
		transform=transforms)
	if limit:
		ds = Subset(ds, list(range(limit)))
		
	loader = DataLoader(ds, batch_size=batch_size,
		shuffle=shuffle,
		pin_memory=True if config.DEVICE == "cuda" else False)
	
	return (ds, loader)