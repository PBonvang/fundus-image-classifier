import torch

from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import BCELoss
from torchvision.models import resnet50
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU, Tanh
from torch.nn import Softmax
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, adam
from torch.optim import Adam

from utils.IModel import IModel
import config

class Network(Module):
    def __init__(self, n_channels):
        super(Network, self).__init__()

        self.conv1 = Conv2d(1,8,3)
        self.conv2 = Conv2d(8,16,3)
        self.conv3 = Conv2d(16,32,3)

        self.lin1 = Linear(32*26*20, 100)
        self.lin2 = Linear(100, 1)

        self.bn1 = BatchNorm2d(8)
        self.bn2 = BatchNorm2d(16)
        self.bn3 = BatchNorm2d(32)

        self.act1 = ReLU()
        self.act2 = Softmax(dim=1)
        self.act3 = Sigmoid()
        self.pool = MaxPool2d((2,2))
        


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act1(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = self.act2(x)
        x = self.lin2(x)
        x = self.act3(x)
        x = torch.squeeze(x)
        

        return x
# END NETWORK DEFINITION

# DEFINE MODEL HERE
class Model(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor([0.35787567893783945]).to(config.DEVICE))
    optimizer_func = Adam
    epochs = 5
    batch_size = 32
    lr = 0.0001

    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
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
    network = Network(3)

    for param in network.parameters():
        param.requires_grad = True

    model = Model(network)
    # END MODEL INSTANTIATION

    return model