import shutil
import config
import create_dataloader
from torchvision import transforms
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Model import get_model
from utils.IModel import IModel
from utils.evaluation import evaluate_model
from utils.ModelMetadata import ModelMetadata
from utils.training import train_model
from utils.validation import model_is_valid

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

# Defining network
model = get_model()

if not model_is_valid(model):
    raise TypeError(
        f"Model is missing attributes. Please define missing attributes and try again. Required attributes are defined in the {IModel.__name__} interface.")

print("[INFO] Loading dataset")
(training_ds, training_dl) = create_dataloader.get_dataloader(
    config.TRAIN,
    transforms=model.training_transforms,
    batch_size=model.batch_size)
(val_ds, val_dl) = create_dataloader.get_dataloader(
    config.VAL,
    transforms=model.validation_transforms,
    batch_size=model.batch_size, shuffle=False)
print("[INFO] Dataset loaded succesfully\n")

print("[INFO] Training model")
train_model(model, training_dl, val_dl, tb_writer)
print("[INFO] Training finished\n")

print("[INFO] Evaluating model")
acc = evaluate_model(model, val_dl)*100
print(f'Accuracy: {acc:.5f} %')
print("[INFO] Evaluation finished\n")

print("[INFO] Saving model")
metadata = ModelMetadata(model, acc)
torch.save(model.network.state_dict(), metadata.model_path)

# Copy model blueprint
shutil.copy(config.MODEL_FILE, metadata.class_path)

# Add metadata to model info file
if not os.path.exists(config.MODELS_INFO_FILE_PATH):
    with open(config.MODELS_INFO_FILE_PATH, "w") as info_file:
        header = ",".join(ModelMetadata.serialization_attributes)
        info_file.write(f"{header}\n")
        info_file.write(str(metadata))
else:
    with open(config.MODELS_INFO_FILE_PATH, "a") as info_file:
        info_file.write(f"\n{metadata}")

print("[INFO] Saved successfully")