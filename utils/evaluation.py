
import torch

from utils.IModel import IModel
import config

def evaluate_model(model: IModel, val_dl):
    network = model.network
    network.eval()

    n_correct_predictions = 0
    for (inputs, targets) in val_dl:
        inputs = inputs.to(config.DEVICE).float()
        targets = targets.to(config.DEVICE).float()
        # evaluate the model on the test set
        output = network(inputs)
        
        
        n_correct_predictions += get_sum_of_correct_predictions(output, targets)
    
    
    acc = n_correct_predictions/len(val_dl.dataset)
    return acc

def get_sum_of_correct_predictions(outputs, labels):
    return (torch.round(outputs) == labels).sum().detach().item()