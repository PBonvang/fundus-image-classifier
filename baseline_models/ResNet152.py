import torch

import torchvision.models as models
from torchvision import transforms
from torch import nn

from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from utils.IModel import IModel
import config

# DEFINE NETWORK HERE:
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        pretrained_model = models.resnet152(pretrained=True)
        self.features = nn.ModuleList(pretrained_model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        
        in_features = pretrained_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
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
    loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))
    optimizer_func = Adam
    epochs = 50
    batch_size = 32
    lr = 0.00001

    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
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
            self.network.fc.parameters(),
            lr=self.lr)
# END MODEL DEFINITION


def get_model() -> IModel:
    # INSTANTIATE MODEL HERE:
    network = Network()
    model = Model(network)
    # END MODEL INSTANTIATION

    return model
