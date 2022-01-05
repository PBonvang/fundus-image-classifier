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
from sys import getsizeof

from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# model definition
class CNN(Module):
    # define model elements
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        #kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2,2), stride=(2,2))

        # second hidden layer
        # self.hidden2 = Conv2d(32, 32, (3,3))
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # self.act2 = ReLU()
        # # second pooling layer
        # self.pool2 = MaxPool2d((2,2), stride=(2,2))
        
        # fully connected layer
        self.hidden3 = Linear(127*127*32, 100)
        #kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(100, 2)
        #xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)

        # second hidden layer
        # X = self.hidden2(X)
        # X = self.act2(X)
        # X = self.pool2(X)

        # flatten
        X = torch.flatten(X,1)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config.LR, momentum=0.9)

    n_steps = len(train_dl)

    # enumerate epochs
    for epoch in range(config.EPOCHS):
        e_start = time.perf_counter()

        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            print(f"Epoch: [{epoch+1}/{config.EPOCHS}], Step: [{i+1}/{n_steps}] Loss: {loss.detach().item():.5f}")
        
        print(f"Epoch: [{epoch+1}/{config.EPOCHS}], Time: {time.perf_counter() - e_start:.2f}s")

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

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
	batch_size=512, shuffle=False)

# Defining network
model = CNN(1)

print("[INFO] Training model")
train_model(training_dl, model)

print("[INFO] Evaluating model")
acc = evaluate_model(val_dl, model)
print(f'Accuracy: {acc*100:.5f} %')

torch.save(model.state_dict(), config.SAVE_PATH)