from datetime import datetime
import logging
import sys
import time
import shutil
import os

from torch.utils.data.dataloader import DataLoader
from utils.RunInfo import RunInfo
import utils.dataloading as dataloading
import torch
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna import Trial
from optuna.trial import TrialState
from optuna.study import StudyDirection

from HyperModel import HyperModel
import config
from utils.training import train_one_epoch
from utils.evaluation import get_sum_of_correct_predictions
from display_study import print_top_5_trials

# Configuration
N_VALID_BATCHES = 1000
N_TRIALS = 1 # number of wanted trials in study
TIME_OUT = None # sec
STUDY_NAME = "test_study"

def objective(trial: Trial):
    trial_start = time.perf_counter()
    print(f"Trial: [{trial.number +1}/{N_TRIALS}]")

    model = HyperModel(trial)
    network = model.network.to(config.DEVICE)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(
        os.path.join(config.STUDIES_PATH, trial.study.study_name,"tensorBoard",f"Trial{trial.number}_{timestamp}"))

    if config.DEBUG:
        print(f"""
    Parameters:
        {trial.params}

    Network:
        {network}
        """)

    training_ds = dataloading.get_dataset(
        config.TRAIN_INFO, config.TRAIN, transforms=model.training_transforms)

    run_info = RunInfo(
        model=model,
        run_path="",
        ds_size=len(training_ds)
    )

    for epoch in range(model.epochs):
        if config.DEBUG:
            print(f"    Epoch [{epoch+1}/{model.epochs}]")
        network.train()

        train_dl = DataLoader(
            training_ds,
            batch_size=model.batch_size,
            shuffle=True
        )
        avg_loss, avg_acc = train_one_epoch(model, train_dl, epoch, run_info, tb_writer)
        trial.report(avg_loss, epoch)
        tb_writer.add_scalar('Epoch/Training loss', avg_loss, epoch)
        tb_writer.add_scalar('Epoch/Training accuracy', avg_acc, epoch)
        tb_writer.close()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Evaluate model
    network.eval()

    test_ds = dataloading.get_dataset(
        config.TEST_INFO, config.TEST, transforms=model.validation_transforms)

    test_dl = DataLoader(
        test_ds,
        batch_size=model.batch_size,
        shuffle=True
    )
    val_loss = []
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dl):
            # Limiting validation data.
            if batch_idx == N_VALID_BATCHES:
                break
            data, target = data.to(config.DEVICE), target.to(
                config.DEVICE).float()
            output = network(data)

            loss = model.loss_func(output, target)
            val_loss.append(loss.detach().item())

            correct += get_sum_of_correct_predictions(output, target)

    avg_loss = sum(val_loss)/len(val_loss)
    accuracy = correct / min(len(test_dl.dataset), model.batch_size*N_VALID_BATCHES)
    trial.set_user_attr("Accuracy", accuracy)

    model_save_path = os.path.join(config.STUDIES_PATH, trial.study.study_name, "models",f"Trial{trial.number}_{timestamp}.pth")
    torch.save(network.state_dict(), model_save_path)

    if config.DEBUG:
        print(f"Accuracy: {accuracy*100:.5f}%")

    print(f"Trial execution time: {(time.perf_counter() - trial_start)/60:.2f} min")
    return avg_loss


if __name__ == "__main__":
    study_path = os.path.join(config.STUDIES_PATH, STUDY_NAME)
    if not os.path.exists(study_path):
        os.makedirs(study_path)
        os.makedirs(os.path.join(study_path, "models"))

    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))
    storage_name = f"sqlite:///{config.STUDIES_PATH}/{STUDY_NAME}/{STUDY_NAME}.db"

    study = optuna.create_study(study_name=STUDY_NAME, storage=storage_name,
                                direction=StudyDirection.MINIMIZE, load_if_exists=True)

    blueprint_dest = os.path.join(config.HYPER_MODELS_PATH, f"{STUDY_NAME}.py")
    if not os.path.exists(blueprint_dest):
        shutil.copy(config.HYPER_MODEL_DEF, blueprint_dest)
    
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, timeout=TIME_OUT)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Top trials:")
    print_top_5_trials(study)
