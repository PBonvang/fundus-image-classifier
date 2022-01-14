from datetime import datetime
import logging
import sys
import time

import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import utils.dataloading as dataloading
import torch
import os
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna import Trial
from optuna.trial import TrialState
from optuna.study import StudyDirection

from HyperModel import HyperModel
import config
from utils.training import train_one_epoch
from utils.evaluation import conv_v

if not os.path.exists(config.STUDIES_PATH):
    os.makedirs(config.STUDIES_PATH)

# Configuration
EPOCHS = 50
BATCH_SIZE = 64
N_VALID_EXAMPLES = 1000
N_TRIALS = 10

optuna.logging.get_logger("optuna").addHandler(
    logging.StreamHandler(sys.stdout))
study_name = "50epoch-64bs"
storage_name = f"sqlite:///{config.STUDIES_PATH}/{study_name}.db"

trial_n = 0
def objective(trial: Trial):
    trial_start = time.perf_counter()
    global trial_n
    print(f"Trial: [{trial_n+1}/{N_TRIALS}]")
    trial_n += 1
    
    model = HyperModel(trial)
    network = model.network
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(f'runs/studies/{study_name}/Trial{trial_n}_{timestamp}')
    
    if config.DEBUG:
        print(f"""
    Parameters:
        {trial.params}

    Network:
        {network}
        """)

    training_ds = dataloading.get_dataset(
        config.TRAIN_INFO, config.TRAIN, transforms=model.training_transforms)

    for epoch in range(EPOCHS):
        if config.DEBUG:
            print(f"    Epoch [{epoch+1}/{EPOCHS}]")
        network.train()
        
        train_dl = DataLoader(
            training_ds,
            batch_size=BATCH_SIZE
        )
        avg_loss = train_one_epoch(model, train_dl, epoch, tb_writer)
        trial.report(avg_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Evaluate model
    network.eval()

    test_ds = dataloading.get_dataset(
        config.TEST_INFO, config.TEST, transforms=model.validation_transforms)

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loss = []
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dl):
            # Limiting validation data.
            if batch_idx == N_VALID_EXAMPLES:
                break
            data, target = data.to(config.DEVICE), target.to(config.DEVICE).float()
            output = network(data)
            
            loss = model.loss_func(output, target)
            val_loss.append(loss.detach().item())

            pred = output.cpu()
            target = target.cpu()
            pred = conv_v(pred)
            correct += (pred == target.numpy()).sum().item()

    avg_loss = sum(val_loss)/len(val_loss)
    accuracy = correct / min(len(test_dl.dataset), BATCH_SIZE*N_VALID_EXAMPLES)
    
    tb_writer.add_scalar("Avg. loss", avg_loss)
    tb_writer.add_scalar("Accuracy", accuracy)
    trial.set_user_attr("Accuracy", accuracy)

    print(f"Trial execution time: {time.perf_counter() - trial_start:.2}s")
    return avg_loss


if __name__ == "__main__":
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                direction=StudyDirection.MINIMIZE, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Top trials:")
    complete_trials.sort(key=lambda t: t.value)

    top_five = complete_trials[:5]
    for i, trial in enumerate(top_five):
        print(f"\nTrial {i+1}:")
        acc = float(trial.user_attrs["Accuracy"])*100
        print("  Value: ", trial.value)
        print(f"  Accuracy: {acc}")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
