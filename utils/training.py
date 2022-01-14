import config
import torch
import time
import os
import math
import torchvision
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from datetime import datetime
import sys

from utils.IModel import IModel


def train_model(
        model: IModel,
        train_dl,
        val_dl,
        tb_writer):
    best_vloss = 1_000_000.
    best_net = None
    example_images, example_labels = next(iter(train_dl))
    tb_writer.add_image('Example fundus pictures', torchvision.utils.make_grid(example_images))
    tb_writer.add_graph(model.network.to('cpu'), example_images)
    for epoch in range(model.epochs):
        e_start = time.perf_counter()
        print(f'Epoch: [{epoch+1}/{model.epochs}]')

        # Make sure gradient tracking is on, and do a pass over the data
        model.network.train(True)
        avg_loss = train_one_epoch(
            model, train_dl, epoch, tb_writer)
        # We don't need gradients on to do reporting
        model.network.eval()

        running_vloss = 0.0
        for (vinputs, vlabels) in val_dl:
            vinputs = vinputs.to(config.DEVICE).float()
            vlabels = vlabels.to(config.DEVICE).float()
            voutputs = model.network(vinputs)
            vloss = model.loss_func(voutputs, vlabels)
            running_vloss += vloss.detach().item()

        avg_vloss = running_vloss / len(val_dl)
        print(
            f'Training loss: {avg_loss:.5f}, Validation loss: {avg_vloss:.5f}, Time: {time.perf_counter() - e_start:.2f}s')

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalar('Loss epoch/train', avg_loss, epoch+1)
        tb_writer.add_scalar('Loss epoch/test', avg_vloss, epoch+1)
        tb_writer.close()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_net = model.network.state_dict()

    model.network.load_state_dict(best_net)


def train_one_epoch(
        model: IModel,
        train_dl,
        epoch_index,
        tb_writer):
    running_loss = 0.
    last_loss = 0.
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

        running_correct += (torch.round(F.sigmoid(outputs)) == labels).sum().detach().item()
        print(f"Step: [{i+1}/{n_steps}]")

        if (i+1) % writer_precision == 0:
            print(running_correct)
            last_loss = running_loss / writer_precision  # loss per batch
            last_correct = running_correct / (writer_precision * len(labels))
            print(f'  Batch: [{i+1}/{n_steps}], Loss: {last_loss:.5f}')
            tb_x = epoch_index * n_steps + i + 1
            tb_writer.add_scalar('Training loss', last_loss, tb_x)
            tb_writer.add_scalar('Training accuracy', last_correct, tb_x)
            tb_writer.close()
            running_loss = 0.
            running_correct = 0

    return last_loss
