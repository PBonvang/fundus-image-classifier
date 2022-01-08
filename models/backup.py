import torch

from imutils import paths
from torch.nn.modules.activation import Sigmoid
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.optim import Adam

from utils.IModel import IModel
import config


# DEFINE NETWORK HERE:
class Network(IModel):
    def __init__(self, n_channels):
        super(Network, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2,2), stride=(2,2))

        # second hidden layer
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        
        # fully connected layer
        self.hidden3 = Linear(62*62*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        
        # output layer
        self.hidden4 = Linear(100, 2) # Use one output bc binary
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1) # Use sigmoid
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)

        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)

        # flatten
        X = torch.flatten(X,1)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X
# END NETWORK DEFINITION


def get_model() -> IModel:
    # INSTANTIATE NETWORK HERE:
    model = Network(1)
    # END NETWORK INSTANTIATION

    # SET MODEL ATTRIBUTES HERE:
    test = BCEWithLogitsLoss()
    model.loss_f = BCEWithLogitsLoss()
    model.loss_f = test
    model.optimizer_func = Adam
    model.epochs = 1
    model.batch_size = 128
    model.lr = 0.001
    # END MODEL ATTRIBUTES

    # DEFINE OPTIMIZER HERE:
    model.optimizer = model.optimizer_func(
        model.parameters(),
        lr=model.lr)
    # END OPTIMIZER DEFINITION

    # DEFINE TRANSFORMS HERE:
    model.training_tansforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    model.validation_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    # END TRANSFORMS

    return model