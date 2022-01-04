from torch.utils import data
import config
import create_dataloader
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

training_tansform = transforms.Compose([
	transforms.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])
validation_transform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# create data loaders
(training_ds, training_loader) = create_dataloader.get_dataloader(config.TRAIN,
	transforms=training_tansform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE)
(val_ds, val_loader) = create_dataloader.get_dataloader(config.VAL,
	transforms=validation_transform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)