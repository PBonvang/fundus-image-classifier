import torch

from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from torchvision.models.resnet import resnet18
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

LOSS_FUNC = BCEWithLogitsLoss()
OPTIMIZER_FUNC = Adam

def get_tl_model(n_classes):
    model = resnet18(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    model_output_feats = model.fc.in_features
    model.fc = Linear(model_output_feats, 2)

    return model
