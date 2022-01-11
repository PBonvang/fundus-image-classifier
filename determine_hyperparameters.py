import logging
import sys
import shutil

import numpy as np
from torch import nn
import config
from utils.dataloading import get_super_dataloader, get_sample_dataloader
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import optuna
from optuna import Trial
from optuna.trial import TrialState
from optuna.study import StudyDirection

from HyperModel import HyperModel
from utils.IModel import IModel
from utils.evaluation import evaluate_model
from utils.ModelMetadata import ModelMetadata
from utils.training import train_model
from utils.validation import model_is_valid

EPOCHS = 10
DS_WEIGHT = 3.492063492
BATCH_SIZE = 16

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "resnet-replica-study"
storage_name = f"sqlite:///{study_name}.db"

def get_dataloaders(model: IModel):
    training_ds = train_dl = val_ds = val_dl = None
    if config.ON_SUPER_COM:
        (training_ds, train_dl) = get_super_dataloader(
            config.TRAIN_INFO,
            config.TRAIN,
            transforms=model.training_transforms,
            batch_size=BATCH_SIZE)
        (val_ds, val_dl) = get_super_dataloader(
            config.VAL_INFO,
            config.VAL,
            transforms=model.validation_transforms,
            batch_size=BATCH_SIZE)
    else:
        (training_ds, train_dl) = get_sample_dataloader(
            config.TRAIN,
            transforms=model.training_transforms,
            batch_size=BATCH_SIZE)
        (val_ds, val_dl) = get_sample_dataloader(
            config.VAL,
            transforms=model.validation_transforms,
            batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dl, val_dl

def objective(trial: Trial):
    model = HyperModel(trial)
    network = model.network

    train_dl, val_dl = get_dataloaders(model)
    N_TRAIN_EXAMPLES = 30
    N_VALID_EXAMPLES = 10

    # Training of the model.
    for epoch in range(EPOCHS):
        network.train()
        for batch_idx, (data, target) in enumerate(train_dl):
            # Limiting training data for faster epochs.
            if batch_idx >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(config.DEVICE), target.to(config.DEVICE).float()

            model.optimizer.zero_grad()
            output = network(data)

            loss = model.loss_func(output, target)
            loss.backward()
            model.optimizer.step()

        # Validation of the model.
        network.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dl):
                # Limiting validation data.
                if batch_idx >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(config.DEVICE), target.to(config.DEVICE)
                output = network(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(val_dl.dataset), BATCH_SIZE*N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction=StudyDirection.MAXIMIZE, load_if_exists=True)
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
