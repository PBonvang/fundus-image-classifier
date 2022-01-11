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
from torch.optim import SGD, adam
from torch.optim import Adam

from utils.IModel import IModel
import config

# DEFINE NETWORK HERE:


# class Network(Module):
#     def __init__(self, n_channels):
#         super(Network, self).__init__()

#         # input to first hidden layer
#         self.hidden1 = Conv2d(n_channels, 32, (3,3))
#         kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         self.act1 = ReLU()
#         # first pooling layer
#         self.pool1 = MaxPool2d((2,2), stride=(2,2))

#         # second hidden layer
#         self.hidden2 = Conv2d(32, 32, (3,3))
#         kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         self.act2 = ReLU()
#         # second pooling layer
#         self.pool2 = MaxPool2d((2,2), stride=(2,2))
        
#         # fully connected layer
#         self.hidden3 = Linear(62*62*32, 100)
#         kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
#         self.act3 = ReLU()
        
#         # output layer
#         self.hidden4 = Linear(100, 1)
#         xavier_uniform_(self.hidden4.weight)
#         self.act4 = Sigmoid()

#     # forward propagate input
#     def forward(self, x):
#         # input to first hidden layer
#         x = self.hidden1(x)
#         x = self.act1(x)
#         x = self.pool1(x)

#         # second hidden layer
#         x = self.hidden2(x)
#         x = self.act2(x)
#         x = self.pool2(x)

#         # flatten
#         x = torch.flatten(x,1)
#         # third hidden layer
#         x = self.hidden3(x)
#         x = self.act3(x)
#         # output layer
#         x = self.hidden4(x)
#         x = self.act4(x)
#         x = torch.squeeze(x)
#         return x
# # END NETWORK DEFINITION

# # DEFINE MODEL HERE
# class Model(IModel):
#     # SET MODEL ATTRIBUTES HERE:
#     loss_func = BCEWithLogitsLoss()
#     optimizer_func = Adam
#     epochs = 5
#     batch_size = 16
#     lr = 0.001

#     training_transforms = transforms.Compose([
#         transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
#         transforms.Grayscale(),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(90),
#         transforms.ToTensor(),
#         #transforms.Normalize(mean=config.MEAN, std=config.STD)
#     ])

#     validation_transforms = transforms.Compose([
#         transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         #transforms.Normalize(mean=config.MEAN, std=config.STD)
#     ])
#     # END MODEL ATTRIBUTES

#     def __init__(self, network):
#         super(Model, self).__init__()

#         self.network = network
#         self.optimizer = self.optimizer_func(
#             self.network.parameters(),
#             lr=self.lr)
# # END MODEL DEFINITION


# def get_model() -> IModel:
#     # INSTANTIATE MODEL HERE:
#     network = Network(1) 
#     model = Model(network)
#     # END MODEL INSTANTIATION

#     return model

class Network(Module):
    def __init__(self, n_channels):
        super(Network, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.features = nn.ModuleList(resnet18.children())[:-1]
        # Now we have our layers up to the fc layer, but we are not finished yet
        # we need to feed these to nn.Sequential() as well, this is needed because,
        # nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). Therefore we use
        # nn.Sequential and since sequential doesnt accept lists, we
        # unpack all the items and send them like this
        self.features = nn.Sequential(*self.features)
        # now lets add our new layers
        in_features = resnet18.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1)
        )

    # forward propagate input
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        x = torch.flatten(x)
        return x
# END NETWORK DEFINITION

# DEFINE MODEL HERE


class Model(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss()
    optimizer_func = Adam
    epochs = 2
    batch_size = 128
    lr = 0.001

    training_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        # transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        # transforms.Grayscale(),
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
        param.requires_grad = False

    for param in network.fc.parameters():
        param.requires_grad = True

    model = Model(network)
    # END MODEL INSTANTIATION

    return model