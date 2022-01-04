import config
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import os

def get_dataloader(data_dir, transforms, batch_size, shuffle=True, limit=0):
	ds = datasets.ImageFolder(root=data_dir,
		transform=transforms)
	if limit:
		ds = Subset(ds, list(range(limit)))
		
	loader = DataLoader(ds, batch_size=batch_size,
		shuffle=shuffle,
		num_workers=os.cpu_count(),
		pin_memory=True if config.DEVICE == "cuda" else False)
	
	return (ds, loader)