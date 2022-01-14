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

from torch.nn import Conv2d, BatchNorm2d, Flatten, Sequential
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
import torch.optim
import optuna
from optuna import Trial

from utils.IModel import IModel
import config
from utils.model import conv_output_shape

# DEFINE NETWORK HERE:

class Network(Module):
    def __init__(self, trial: Trial):
        super(Network, self).__init__()

        self.blur = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, stride=1)
        self.blur.weight = nn.Parameter(torch.ones((1,1,3,3))/9.0)

        conv_layers = []

        in_shape = config.IMAGE_SHAPE
        layer_1_features = trial.suggest_categorical("n_layer_1_features", [4,8,16,32,64])
        # Layer 1
        conv_layers.extend([
            Conv2d(1,layer_1_features,3),
            BatchNorm2d(layer_1_features),
            ReLU(),
            MaxPool2d((2,2))
        ])

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        layer_2_features = trial.suggest_categorical("n_layer_2_features", [16,32,64,128,256])
        # Layer 2
        conv_layers.extend([
            Conv2d(layer_1_features,layer_2_features,3),
            BatchNorm2d(layer_2_features),
            ReLU(),
            MaxPool2d((2,2))
        ])


        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))


        layer_3_features = trial.suggest_categorical("n_layer_3_features", [32,64,128,256,512])
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
        layer_4_features = trial.suggest_int("n_layer_4_features", 80, 200 )

        # Layer 4
        lin_layers.extend([
            Flatten(),
            Linear(in_features, layer_4_features),
            Linear(layer_4_features,1),
            Sigmoid()
        ])

        self.lin_layers = Sequential(*lin_layers)
        self.layers = Sequential(
            self.conv_layers,
            self.lin_layers
        )

    def forward(self, x):
        x = self.blur(x)
        x = self.conv_layers(x)
        x = self.lin_layers(x)
        x = torch.squeeze(x)

        return x

    def reset_weights(self):
        for layer in self.layers.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# END NETWORK DEFINITION

# DEFINE MODEL HERE
class HyperModel(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))

    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.GaussianBlur(3),
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

    def __init__(self, trial):
        super(HyperModel, self).__init__()
        network = Network(trial).to(config.DEVICE)

        self.network = network

        optimizer_name = trial.suggest_categorical(
            'optimizer', ['Adam', 'RMSprop', 'SGD'])
        self.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.network.layers.parameters(), lr=self.lr)

# END MODEL DEFINITION