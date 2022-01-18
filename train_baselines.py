from datetime import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from baseline_models import Densenet201, GNet, ResNet101, ResNet152
from utils.RunInfo import RunInfo
import utils.dataloading as dataloading
import config
from utils.evaluation import evaluate_model
from utils.training import train_one_epoch


EPOCHS = 1

base_line_info_file = os.path.join("baselines","baselines.csv")
baselines = [Densenet201, GNet, ResNet101, ResNet152]
models = [b.get_model() for b in baselines]

training_ds = dataloading.get_dataset(
        config.TRAIN_INFO, config.TRAIN, transforms=models[0].training_transforms)

for i, model in enumerate(models):
    model_name = baselines[i].__name__.split(".")[-1]
    print("Baseline model: ", model_name)
    print("------------------------------------")

    run_path = os.path.join("baselines", model_name)
    network = model.network
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(
        os.path.join(run_path,"tensorboard", timestamp))
        
    run_info = RunInfo(
        model=model,
        name=model_name,
        run_path=run_path,
        ds_size=len(training_ds)
    )

    for epoch in range(EPOCHS):
        if config.DEBUG:
            print(f"    Epoch [{epoch+1}/{EPOCHS}]")
        network.train()

        train_dl = DataLoader(
            training_ds,
            batch_size=model.batch_size,
            shuffle=True
        )
        avg_loss, avg_acc = train_one_epoch(model, train_dl, epoch, run_info, tb_writer)
        tb_writer.add_scalar('Epoch/Training loss', avg_loss, epoch)
        tb_writer.add_scalar('Epoch/Training accuracy', avg_acc, epoch)
        tb_writer.close()
    
    network.eval()

    test_ds = dataloading.get_dataset(
        config.TEST_INFO, config.TEST, transforms=model.validation_transforms)

    test_dl = DataLoader(
        test_ds,
        batch_size=model.batch_size,
        shuffle=True
    )
    
    acc =  evaluate_model(model, test_dl, run_path)*100

    run_info.set_test_accuracy(acc)
    run_info.save_to_csv(base_line_info_file)

    model_save_path = os.path.join(run_path,f"{timestamp}.pth")
    torch.save(network.state_dict(), model_save_path)

    print(f"    Accuracy: {acc:.5f} %")
