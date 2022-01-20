import torch

from torchvision import transforms
from torch import nn
import numpy as np

from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU, Sequential, Flatten, Sigmoid
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from utils.IModel import IModel
import config
from utils.model import conv_output_shape

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        self.blur = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, stride=1)
        self.blur.weight = nn.Parameter(torch.ones((1,1,3,3))/9.0)
        self.blur.weight.requires_grad = False

        conv_layers = []

        in_shape = config.IMAGE_SHAPE
        layer_1_features = 64
        # Layer 1
        conv_layers.extend([
            Conv2d(1,layer_1_features,3),
            BatchNorm2d(layer_1_features),
            ReLU(),
            MaxPool2d((2,2))
        ])

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        layer_2_features = 128
        # Layer 2
        conv_layers.extend([
            Conv2d(layer_1_features,layer_2_features,3),
            BatchNorm2d(layer_2_features),
            ReLU(),
            MaxPool2d((2,2))
        ])

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        layer_3_features = 256
        # Layer 3
        conv_layers.extend([
            Conv2d(layer_2_features,layer_3_features,3),
            BatchNorm2d(layer_3_features),
            ReLU(),
            MaxPool2d((2,2))
        ])

        self.conv_layers = Sequential(*conv_layers)
        lin_layers = []

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        in_features = int(np.prod(in_shape))*layer_3_features
        layer_4_features = 100

        # Layer 4
        lin_layers.extend([
            Flatten(),
            Linear(in_features, layer_4_features),
            Linear(layer_4_features,1),
            Sigmoid()
        ])

        self.lin_layers = Sequential(*lin_layers)

    def forward(self, x):
        x = self.blur(x)
        x = self.conv_layers(x)
        x = self.lin_layers(x)
        x = torch.squeeze(x)

        return x

# END NETWORK DEFINITION

# DEFINE MODEL HERE
class Model(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))
    optimizer_func = Adam
    epochs = 500
    batch_size = 32
    lr = 0.0000022

    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.GaussianBlur(3)
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.ToTensor(),
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
    network = Network()

    for param in network.parameters():
        param.requires_grad = True

    for param in network.blur.parameters():
        param.requires_grad = False

    model = Model(network)
    # END MODEL INSTANTIATION

    return model