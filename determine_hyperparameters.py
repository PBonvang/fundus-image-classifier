import logging
import sys

import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import utils.dataloading as dataloading
import torch
import os

import optuna
from optuna import Trial
from optuna.trial import TrialState
from optuna.study import StudyDirection

from HyperModel import HyperModel
from sklearn.model_selection import KFold
import config
from utils.evaluation import conv_v

if not os.path.exists(config.STUDIES_PATH):
    os.makedirs(config.STUDIES_PATH)

# Configuration
KFOLDS = 5
EPOCHS = 1
BATCH_SIZE = 64
N_TRAIN_EXAMPLES = 1
N_VALID_EXAMPLES = 1
N_TRIALS = 1
DEBUG = False

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "test_study"
storage_name = f"sqlite:///{config.STUDIES_PATH}/{study_name}.db"

trial_n = 0
def objective(trial: Trial):
    global trial_n
    print(f"Trial: [{trial_n}/{N_TRIALS-1}]")
    trial_n += 1
    model = HyperModel(trial)
    network = model.network

    training_ds = dataloading.get_dataset(config.TRAIN_INFO, config.TRAIN, transforms=model.training_transforms)

    kfold = KFold(n_splits=KFOLDS, shuffle=True)
    accuracies = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(training_ds)):
        if DEBUG:
            print(f'Fold [{fold+1}/{KFOLDS}]')
            print('--------------------------------')
        network.reset_weights()

        # Setup dataloaders
        train_dl = DataLoader(
            training_ds,
            batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(train_ids)
        )
        val_dl = DataLoader(
            training_ds,
            batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(val_ids)
        )

        # Training of the model.
        for epoch in range(EPOCHS):
            if DEBUG: print(f"    Epoch [{epoch+1}/{EPOCHS}]")
            network.train()
            for batch_idx, (data, target) in enumerate(train_dl):
                # Limiting training data for faster epochs.
                if batch_idx == N_TRAIN_EXAMPLES:
                    break
                if DEBUG: print(f"        Step [{batch_idx+1}/{len(train_dl)}]")

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
                if batch_idx == N_VALID_EXAMPLES:
                    break
                data, target = data.to(config.DEVICE), target.cpu()
                output = network(data)
                # Get the index of the max log-probability.
                pred = output.cpu()
                pred = conv_v(pred)
                correct += (pred == target.numpy()).sum().item()

        accuracy = correct / min(len(val_dl.dataset), BATCH_SIZE*N_VALID_EXAMPLES)
        accuracies.append(accuracy)

        trial.report(accuracy, fold)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_acc = np.mean(accuracies)
    return avg_acc

if __name__ == "__main__":
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction=StudyDirection.MAXIMIZE, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Top trials:")
    complete_trials.sort(key=lambda t: t.value)
    complete_trials.reverse()

    top_five = complete_trials[:5]
    for i, trial in enumerate(top_five):
        print(f"\nTrial {i+1}:")
        print("  Value: ", trial.value)
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
