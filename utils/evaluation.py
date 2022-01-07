
from numpy import array, vectorize, vstack
from numpy import argmax
import numpy
from sklearn.metrics import accuracy_score

from utils.IModel import IModel

def convert_to_class_labels(output: array):
    return  1 if output >= 0.5 else 0

conv_v = vectorize(convert_to_class_labels)


def evaluate_model(model: IModel, val_dl):
    network = model.network
    network.eval()
    network.cpu()
    predictions, actuals = list(), list()

    for (inputs, targets) in val_dl:
        inputs = inputs.float()
        targets = targets.float()
        # evaluate the model on the test set
        yhat = network(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = conv_v(yhat)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc