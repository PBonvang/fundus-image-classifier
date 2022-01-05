import shutil
import config
import create_dataloader
from torchvision import transforms
import torch
import time
import os

from Model import Model
from utils.evaluation import evaluate_model
from utils.ModelMetadata import ModelMetadata

def train_model(train_dl, model: Model):
    # define the optimization
    optimizer = model.optimizer_func(
        model.parameters(), lr=config.LR, momentum=0.9)

    model = model.to(config.DEVICE)
    n_steps = len(train_dl)

    for epoch in range(config.EPOCHS):
        e_start = time.perf_counter()

        # Enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = model.loss_func(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            print(
                f"Epoch: [{epoch+1}/{config.EPOCHS}], Step: [{i+1}/{n_steps}] Loss: {loss.detach().item():.5f}")

        print(
            f"Epoch: [{epoch+1}/{config.EPOCHS}], Time: {time.perf_counter() - e_start:.2f}s")


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

print("[INFO] Dataset loaded succesfully")

# Defining network
model = Model(1)

print("[INFO] Training model")
train_model(training_dl, model)
print("[INFO] Training finished\n")

print("[INFO] Evaluating model")
acc = evaluate_model(val_dl, model)*100
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