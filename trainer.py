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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from Model import CNN

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

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=config.LR, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_dl):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        global model
        model = model.to(config.DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            print(f'  Batch: [{i+1}/{len(training_dl)}], Loss: {last_loss:.5f}')
            tb_x = epoch_index * len(training_dl) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0


best_vloss = 1_000_000.
print("[INFO] Training model")

for epoch in range(config.EPOCHS):
    e_start = time.perf_counter()
    print(f'Epoch: [{epoch+1}/{config.EPOCHS}]')

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    # We don't need gradients on to do reporting
    model.train(False)
    model.cpu()

    running_vloss = 0.0
    for i, vdata in enumerate(val_dl):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'Training loss: {avg_loss:.5f}, Validation loss: {avg_vloss:.5f}, Time: {time.perf_counter() - e_start:.2f}s')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print("[INFO] Evaluating model")
acc = evaluate_model(val_dl, model)
print(f'Training accuracy: {acc*100:.5f} %')
