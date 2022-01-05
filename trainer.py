from torch.utils import data
import config
import create_dataloader
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import uuid
import os

from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from Model import CNN

CRITERION = CrossEntropyLoss()
OPTIMIZER_F = SGD

# train the model
def train_model(train_dl, model):
    # define the optimization
    optimizer = OPTIMIZER_F(model.parameters(), lr=config.LR, momentum=0.9)

    n_steps = len(train_dl)

    for epoch in range(config.EPOCHS):
        e_start = time.perf_counter()

        # Enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            model = model.to(config.DEVICE)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = CRITERION(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            print(f"Epoch: [{epoch+1}/{config.EPOCHS}], Step: [{i+1}/{n_steps}] Loss: {loss.detach().item():.5f}")
        
        print(f"Epoch: [{epoch+1}/{config.EPOCHS}], Time: {time.perf_counter() - e_start:.2f}s")

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        inputs = inputs.to('cpu')
        targets = targets.to('cpu')
        model = model.to('cpu')

        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

training_tansforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])
validation_transforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# Creating data loaders
(training_ds, training_dl) = create_dataloader.get_dataloader(config.TRAIN,
	transforms=training_tansforms,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE)
(val_ds, val_dl) = create_dataloader.get_dataloader(config.VAL,
	transforms=validation_transforms,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

# Defining network
model = CNN(1)

print("[INFO] Training model")
train_model(training_dl, model)

print("[INFO] Evaluating model")
acc = evaluate_model(val_dl, model)
print(f'Accuracy: {acc*100:.5f} %')

model_id = uuid.uuid4()
model_path = os.path.join(config.MODEL_PATH, f"{model_id}.pth")
torch.save(model.state_dict(), model_path)

model_info = [
    model_id,
    model_path,
    acc,
    config.EPOCHS,
    config.LR,
    type(CRITERION).__name__,
    type(OPTIMIZER_F).__name__
    #model
    ]
with open(config.MODEL_INFO_FILE_PATH, "w") as info_file:
    info_file.write("\n")