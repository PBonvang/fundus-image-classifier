import torch

import torchvision.models as models
from torchvision import transforms
from torch import nn

from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from utils.IModel import IModel
import config

# DEFINE NETWORK HERE
class Network(Module):
    def __init__(self, n_channels):
        super(Network, self).__init__()

        resnet = models.resnet50(pretrained=True)
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.features = nn.ModuleList(resnet.children())[:-1]
        # Now we have our layers up to the fc layer, but we are not finished yet
        # we need to feed these to nn.Sequential() as well, this is needed because,
        # nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). Therefore we use
        # nn.Sequential and since sequential doesnt accept lists, we
        # unpack all the items and send them like this
        self.features = nn.Sequential(*self.features)
        # now lets add our new layers
        in_features = resnet.fc.in_features
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
    loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))
    optimizer_func = Adam
    epochs = 10
    batch_size = 16
    lr = 0.001

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
