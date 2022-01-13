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
import torch.optim
import optuna
from optuna import Trial

from utils.IModel import IModel
import config

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)

Creds:
https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

# DEFINE NETWORK HERE:

class Network(Module):
    def __init__(self, trial: Trial):
        super(Network, self).__init__()

        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 7)
        n_lin_layers = trial.suggest_int('n_lin_layers', 0, 3)
        layers = []

        in_features = 1
        in_shape = config.IMAGE_SHAPE
        for i in range(n_conv_layers):
            out_features = trial.suggest_int(f"n_units_conv_l{i}", 10, 128)
            kernal_size = trial.suggest_int(f"kernal_size_conv_l{i}", 1, 7)
            max_pooling = trial.suggest_int(f"max_pooling_conv_l{i}", 0, 1)

            layers.extend([
                nn.Conv2d(in_features, out_features,
                          kernel_size=kernal_size, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()])

            in_shape = conv_output_shape(in_shape, kernal_size, pad=1)
            if max_pooling:
                layers.append(nn.MaxPool2d(2))
                in_shape = (np.ceil(in_shape[0]/2), np.ceil(in_shape[0]/2))

            in_features = out_features

        dp = trial.suggest_float(f'conv_to_lin_dropout', 0.2, 0.5)

        layers.extend([
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(dp)
        ])

        in_shape = (np.ceil(in_shape[0]/4), np.ceil(in_shape[0]/4))

        in_features = int(np.prod(in_shape))*in_features

        for i in range(n_lin_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 64, 1024)
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])

            in_features = out_features

        # Classifier
        layers.extend([
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            print(x.shape, layer)

        x = torch.flatten(x)
        return x
# END NETWORK DEFINITION

# DEFINE MODEL HERE
class HyperModel(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss(
        pos_weight=torch.tensor([3.492063492]).to(config.DEVICE))

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

    def __init__(self, trial):
        super(HyperModel, self).__init__()
        network = Network(trial).to(config.DEVICE)

        self.network = network

        optimizer_name = trial.suggest_categorical(
            'optimizer', ['Adam', 'RMSprop', 'SGD'])
        self.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        self.optimizer = getattr(torch.optim, optimizer_name)(
            network.parameters(), lr=self.lr)

# END MODEL DEFINITION