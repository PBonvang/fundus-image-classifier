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
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, input):
        output = input

        return output
# END NETWORK DEFINITION


def get_model() -> IModel:
    # INSTANTIATE NETWORK HERE:
    model = resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model_output_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(model_output_feats, 1),
        nn.Softmax(dim=1)
    )
    # END NETWORK INSTANTIATION

    # SET MODEL ATTRIBUTES HERE:
    model.loss_func = BCEWithLogitsLoss()
    model.optimizer_func = Adam
    model.epochs = 1
    model.batch_size = 128
    model.lr = 0.001
    # END MODEL ATTRIBUTES

    # DEFINE OPTIMIZER HERE:
    model.optimizer = model.optimizer_func(
        model.fc.parameters(),
        lr=model.lr)
    # END OPTIMIZER DEFINITION

    # DEFINE TRANSFORMS HERE:
    model.training_tansforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        # transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    model.validation_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    # END TRANSFORMS

    return model
