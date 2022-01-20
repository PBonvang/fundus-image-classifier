import torch
from torch import nn
import torch.optim
from torchvision import transforms
import numpy as np

from optuna import Trial

from utils.IModel import IModel
import config
from utils.model import conv_output_shape


class Network(nn.Module):
    def __init__(self, trial: Trial):
        super(Network, self).__init__()
        # DEFINE NETWORK HERE

        ###########################################
        # Filters
        ###########################################
        self.blur = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, stride=1)
        self.blur.weight = nn.Parameter(torch.ones((1,1,3,3))/9.0)

        ###########################################
        # Network layers
        ###########################################
        conv_layers = []

        in_shape = config.IMAGE_SHAPE
        layer_1_features = trial.suggest_categorical("n_layer_1_features", [8,16,32,64])
        # Layer 1
        conv_layers.extend([
            nn.Conv2d(1,layer_1_features,3),
            nn.BatchNorm2d(layer_1_features),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ])

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        layer_2_features = trial.suggest_categorical("n_layer_2_features", [16,32,64,128,256])
        # Layer 2
        conv_layers.extend([
            nn.Conv2d(layer_1_features,layer_2_features,3),
            nn.BatchNorm2d(layer_2_features),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ])


        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))


        layer_3_features = trial.suggest_categorical("n_layer_3_features", [32,64,128,256,512])
        # Layer 3
        conv_layers.extend([
            nn.Conv2d(layer_2_features,layer_3_features,3),
            nn.BatchNorm2d(layer_3_features),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ])

        self.conv_layers = nn.Sequential(*conv_layers)
        lin_layers = []

        in_shape = conv_output_shape(in_shape, 3)
        in_shape = (np.floor(in_shape[0]/2), np.floor(in_shape[1]/2))

        in_features = int(np.prod(in_shape))*layer_3_features
        layer_4_features = trial.suggest_int("n_layer_4_features", 80, 200 )

        # Layer 4
        lin_layers.extend([
            nn.Flatten(),
            nn.Linear(in_features, layer_4_features),
            nn.Linear(layer_4_features,1),
            nn.Sigmoid()
        ])

        ###########################################
        # Classifier
        ###########################################
        self.lin_layers = nn.Sequential(*lin_layers)
        self.layers = nn.Sequential(
            self.conv_layers,
            self.lin_layers
        )

    def forward(self, x):
        # DEFINE NETWORK FORWARD PROPAGATION OF INPUT HERE
        x = self.blur(x)
        x = self.conv_layers(x)
        x = self.lin_layers(x)
        x = torch.squeeze(x)
        # END FORWARD PROPAGATION

        return x

    def reset_weights(self):
        for layer in self.layers.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# END NETWORK DEFINITION

# DEFINE MODEL HERE
class HyperModel(IModel):
    # SET MODEL ATTRIBUTES HERE:
    ###########################################
    # Training attributes
    ###########################################
    batch_size = 32
    epochs = 50
    loss_func = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.DS_WEIGHT]).to(config.DEVICE))

    ###########################################
    # Training preprocessing
    ###########################################
    training_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
    ])

    ###########################################
    # Validation preprocessing
    ###########################################
    validation_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SHAPE),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    # END MODEL ATTRIBUTES

    def __init__(self, trial: Trial):
        super(HyperModel, self).__init__()
        self.network = Network(trial)

        self.lr = trial.suggest_float('lr', 1e-7, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical(
            'optimizer', ['Adam', 'RMSprop', 'SGD','Adagrad','AdamW','Adamax'])
        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.network.layers.parameters(), lr=self.lr)

# END MODEL DEFINITION