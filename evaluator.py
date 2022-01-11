from torchvision import transforms
import config
import torch
import os

from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score

from utils.evaluation import evaluate_model

model = NetworkModel(1)
model_path = os.path.join(config.MODELS_PATH, "05_01_2022__11_28_34.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

validation_transforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SHAPE, config.IMAGE_SHAPE)),
    transforms.Grayscale(),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])

(val_ds, val_dl) = create_dataloader.get_dataloader(config.VAL,
	transforms=validation_transforms,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

accuracy = evaluate_model(val_dl, model)*100
print(f'Accuracy: {accuracy:.5f} %')