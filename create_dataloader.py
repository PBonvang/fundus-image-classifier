import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_dataloader(data_dir, transforms, batch_size, shuffle=True):
	ds = datasets.ImageFolder(root=data_dir,
		transform=transforms)
	loader = DataLoader(ds, batch_size=batch_size,
		shuffle=shuffle,
		num_workers=os.cpu_count(),
		pin_memory=True if config.DEVICE == "cuda" else False)
	
	return (ds, loader)