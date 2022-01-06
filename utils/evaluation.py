
from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score

from Model import Model

def evaluate_model(model:Model, val_dl):
    model.eval()
    model = model.cpu()
    predictions, actuals = list(), list()

    for (inputs, targets) in val_dl:
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
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