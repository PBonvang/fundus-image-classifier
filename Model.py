import torch

from imutils import paths
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import BCELoss
from torchvision.models import resnet50
import torchvision.models as models
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


class Network(Module):
    def __init__(self, n_channels):
        super(Network, self).__init__()

    def forward(self, x):
        
        return x
# END NETWORK DEFINITION

# DEFINE MODEL HERE
class Model(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss()
    optimizer_func = Adam
    epochs = 5
    batch_size = 16
    lr = 0.001

    training_transforms = transforms.Compose([
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
    # END MODEL ATTRIBUTES

    def __init__(self, network):
        super(Model, self).__init__()

        self.network = network
        self.optimizer = self.optimizer_func(
            self.network.parameters(),
            lr=self.lr)
# END MODEL DEFINITION


def get_model() -> IModel:
    # INSTANTIATE MODEL HERE:
    network = Network(1)
    model = Model(network)
    # END MODEL INSTANTIATION

    return model
