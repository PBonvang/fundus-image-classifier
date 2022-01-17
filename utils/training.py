from datetime import datetime
import os
import config
import torch
import time
import math
import torchvision

from torch.utils.data.dataloader import DataLoader

from utils.IModel import IModel
from utils.RunInfo import RunInfo
from utils.evaluation import get_sum_of_correct_predictions


def train_model(
        model: IModel,
        train_ds,
        test_dl: DataLoader,
        run_info: RunInfo,
        tb_writer):

    example_images, example_labels = next(iter(test_dl))
    tb_writer.add_image('Example fundus pictures',
                        torchvision.utils.make_grid(example_images))
    tb_writer.add_graph(model.network.to('cpu'), example_images)

    for epoch in range(model.epochs):
        e_start = time.perf_counter()
        print(f'Epoch: [{epoch+1}/{model.epochs}]')

        # Make sure gradient tracking is on, and do a pass over the data
        model.network.train(True)
        train_dl = DataLoader(
            train_ds,
            batch_size=model.batch_size,
            shuffle=True
        )
        avg_loss, avg_acc = train_one_epoch(
            model, train_dl, epoch, run_info, tb_writer)
        # We don't need gradients on to do reporting
        model.network.eval()

        running_vloss = 0.0
        running_correct = 0.0
        for (vinputs, vlabels) in test_dl:
            vinputs = vinputs.to(config.DEVICE).float()
            vlabels = vlabels.to(config.DEVICE).float()
            voutputs = model.network(vinputs)
            vloss = model.loss_func(voutputs, vlabels)
            running_vloss += vloss.detach().item()
            running_correct += get_sum_of_correct_predictions(voutputs, vlabels)

        avg_vloss = running_vloss / len(test_dl)
        vacc = running_correct / len(test_dl.dataset)
        print(
            f'Training loss: {avg_loss:.5f}, Validation loss: {avg_vloss:.5f}, Time: {time.perf_counter() - e_start:.2f}s')

        # Log performance metrics for current model
        tb_writer.add_scalar('Epoch performance/Train loss', avg_loss, epoch+1)
        tb_writer.add_scalar('Epoch performance/Train accuracy', avg_acc, epoch+1)
        tb_writer.add_scalar('Epoch performance/Validation loss', avg_vloss, epoch+1)
        tb_writer.add_scalar('Epoch performance/Validation accuracy', vacc, epoch+1)
        tb_writer.close()

        # Track best performance, and save the model's state
        if avg_vloss < run_info.best_validation_loss:
            run_info.best_validation_loss = avg_vloss
            run_info.best_net = model.network.state_dict()
        if vacc > run_info.best_validation_accuracy:
            run_info.best_validation_accuracy = vacc

        # Save model checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        torch.save(model.network.state_dict(),
                   os.path.join(run_info.checkpoint_path, f"E{epoch}_{timestamp}.pth"))

        run_info.epochs_run += 1

    model.network.load_state_dict(run_info.best_net)


def train_one_epoch(
        model: IModel,
        train_dl,
        epoch_index,
        run_info: RunInfo,
        tb_writer):
    running_loss = 0.
    last_loss = 0.
    last_acc = 0.
    running_correct = 0
    n_steps = len(train_dl)
    writer_precision = math.ceil(n_steps/10)
    network = model.network.to(config.DEVICE)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (inputs, labels) in enumerate(train_dl):
        # Every data instance is an input + label pair
        inputs = inputs.to(config.DEVICE).float()
        labels = labels.to(config.DEVICE).float()

        # Zero your gradients for every batch!
        model.optimizer.zero_grad()

        # Make predictions for this batch
        outputs = network(inputs)

        # Compute the loss and its gradients
        loss = model.loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        model.optimizer.step()

        # Gather data and report
        running_loss += loss.detach().item()

        running_correct += get_sum_of_correct_predictions(outputs, labels)
        if config.DEBUG:
            print(f"  Step: [{i+1}/{n_steps}]")

        if (i+1) % writer_precision == 0:
            last_loss = running_loss / writer_precision  # loss per batch
            last_acc = running_correct / (writer_precision * len(labels))

            if config.DEBUG:
                print(f'  Step: [{i+1}/{n_steps}], Loss: {last_loss:.5f}')

            tb_x = epoch_index * n_steps + i + 1
            tb_writer.add_scalar('Performance/Training loss', last_loss, tb_x)
            tb_writer.add_scalar('Performance/Training accuracy', last_acc, tb_x)
            tb_writer.close()

            if last_loss < run_info.best_training_loss:
                run_info.best_training_loss = last_loss
            
            if last_acc > run_info.best_training_accuracy:
                run_info.best_training_accuracy = last_acc

            running_loss = 0.
            running_correct = 0

        run_info.steps_taken += 1
        run_info.samples_seen += len(inputs)

    return last_loss, last_acc
