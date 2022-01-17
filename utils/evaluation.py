
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.IModel import IModel
from utils.FundusDataset import FundusDataset
import config


def evaluate_model(model: IModel, val_dl: FundusDataset):
    network = model.network
    network.eval()

    n_correct_predictions = 0
    predictions, actuals = list(), list()
    for (inputs, targets) in val_dl:
        inputs = inputs.to(config.DEVICE).float()
        targets: torch.Tensor = targets.to(config.DEVICE).float()
        # evaluate the model on the test set
        output = network(inputs)
        
        n_correct_predictions += get_sum_of_correct_predictions(output, targets)

        pred = convert_to_class_labels(output)
        predictions.extend(pred.detach().numpy())
        actuals.extend(targets.detach().numpy())
    
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    plot_confusion_matrix(predictions, actuals)
    
    acc = n_correct_predictions/len(val_dl.dataset)
    return acc

def get_sum_of_correct_predictions(outputs, labels):
    return (torch.round(outputs) == labels).sum().detach().item()

def convert_to_class_labels(output: torch.Tensor) -> torch.Tensor:
    return torch.round(output)

# Creds: https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
def plot_confusion_matrix(predictions, actuals) -> None:
    conf_matrix = confusion_matrix(actuals, predictions)
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    conf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        conf_matrix.flatten()/np.sum(conf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()