import torch

import torchvision.models as models
from torchvision import transforms
import torch
from torch import nn
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss

from utils.IModel import IModel
import config

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        self.blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, stride=1)
        self.blur.weight = nn.Parameter(torch.ones((3,3,3,3))/9.0)
        self.blur.weight.requires_grad = False

        pretrained_model = models.densenet201(pretrained=True)
        self.features = nn.ModuleList(pretrained_model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        
        in_features = pretrained_model.classifier.in_features*8*8
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    # forward propagate input
    def forward(self, x):
        x = self.blur(x)
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        x = torch.flatten(x)
        return x

# END NETWORK DEFINITION

# DEFINE MODEL HERE
class Model(IModel):
    # SET MODEL ATTRIBUTES HERE:
    loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))
    optimizer_func = torch.optim.RMSprop
    epochs = 1
    batch_size = 32
    lr = 0.0000022

    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.GaussianBlur(3)
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
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