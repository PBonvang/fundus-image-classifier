import shutil
import config
import create_dataloader
from torchvision import transforms
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Model import Model
from utils.evaluation import evaluate_model
from utils.ModelMetadata import ModelMetadata

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

def train_one_epoch(model: Model, optimizer, train_dl, epoch_index):
    global tb_writer

    running_loss = 0.
    last_loss = 0.
    n_steps = len(train_dl)
    model = model.to(config.DEVICE)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (inputs, labels) in enumerate(train_dl):
        # Every data instance is an input + label pair
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = model.loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.detach().item()
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            print(f'  Batch: [{i+1}/{n_steps}], Loss: {last_loss:.5f}')
            tb_x = epoch_index * n_steps + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train_model(model: Model, train_dl, val_dl):
    global tb_writer

    best_vloss = 1_000_000.
    best_model = None
    optimizer = model.optimizer_func(
        model.parameters(), lr=config.LR, momentum=0.9)

    for epoch in range(config.EPOCHS):
        e_start = time.perf_counter()
        print(f'Epoch: [{epoch+1}/{config.EPOCHS}]')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, train_dl, epoch)
        # We don't need gradients on to do reporting
        model.train(False)
        model.cpu()

        running_vloss = 0.0
        for (vinputs, vlabels) in val_dl:
            voutputs = model(vinputs)
            vloss = model.loss_func(voutputs, vlabels)
            running_vloss += vloss.detach().item()

        avg_vloss = running_vloss / len(val_dl)
        print(f'Training loss: {avg_loss:.5f}, Validation loss: {avg_vloss:.5f}, Time: {time.perf_counter() - e_start:.2f}s')

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        tb_writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model = model.state_dict()

    return best_model

training_tansforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    #transforms.Normalize(mean=config.MEAN, std=config.STD)
])
validation_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=config.MEAN, std=config.STD)
])

print("[INFO] Loading dataset")
# Creating data loaders
(training_ds, training_dl) = create_dataloader.get_dataloader(config.TRAIN,
                                                              transforms=training_tansforms,
                                                              batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE)
(val_ds, val_dl) = create_dataloader.get_dataloader(config.VAL,
                                                    transforms=validation_transforms,
                                                    batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

print("[INFO] Dataset loaded succesfully\n")

# Defining network
model = Model(1)

print("[INFO] Training model")
best_model = train_model(model, training_dl, val_dl)
model.load_state_dict(best_model)
print("[INFO] Training finished\n")

print("[INFO] Evaluating model")
acc = evaluate_model(model,val_dl)*100
print(f'Accuracy: {acc:.5f} %')
print("[INFO] Evaluation finished\n")

print("[INFO] Saving model")
metadata = ModelMetadata(model, acc, config)
torch.save(model.state_dict(), metadata.model_path)

# Copy model blueprint
shutil.copy(config.MODEL_CLASS, metadata.class_path)

# Add metadata to model info file
if not os.path.exists(config.MODEL_INFO_FILE_PATH):
    with open(config.MODEL_INFO_FILE_PATH, "w") as info_file:
        header = ",".join(ModelMetadata.serialization_attributes)
        info_file.write(f"{header}\n")
        info_file.write(str(metadata))
else:
    with open(config.MODEL_INFO_FILE_PATH, "a") as info_file:
        info_file.write(f"\n{metadata}")

print("[INFO] Saved successfully")