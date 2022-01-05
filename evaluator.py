from torchvision import transforms
import config
import torch
import os

from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score

from Model import CNN
import create_dataloader

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

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

model = CNN(1)
model_path = os.path.join(config.MODEL_PATH, "05_01_2022__11_28_34.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

validation_transforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])

(val_ds, val_dl) = create_dataloader.get_dataloader(config.VAL,
	transforms=validation_transforms,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

accuracy = evaluate_model(val_dl, model)*100
print(f'Accuracy: {accuracy:.5f} %')