import shutil
import config
from utils.dataloading import get_dataset
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

from Model import get_model
from utils.IModel import IModel
from utils.evaluation import evaluate_model
from utils.ModelMetadata import ModelMetadata
from utils.training import train_model
from utils.validation import model_is_valid

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter(f'runs/isFundus{timestamp}')

##################################################################
#                    Model initialization
##################################################################
print("[INFO] Initializing model")
model = get_model()

if not model_is_valid(model):
    raise TypeError(
        f"Model is missing attributes. Please define missing attributes and try again. Required attributes are defined in the {IModel.__name__} interface.")

model_metadata = ModelMetadata(model)

trained_model_path = os.path.join(config.TRAINED_MODELS_PATH, model.id)
os.makedirs(trained_model_path)
os.makedirs(os.path.join(trained_model_path, "checkpoints"))
print("[INFO] Model initialized")

##################################################################
#                    Blueprint saving
##################################################################
print("\n[INFO] Saving model blueprint")
shutil.copy(config.MODEL_DEF, 
    os.path.join(config.MODELS_PATH, f"{model.id}.py"))
print("[INFO] Model blueprint saved")

##################################################################
#                    Load datasets
##################################################################
print("\n[INFO] Loading dataset")
training_ds = get_dataset(config.TRAIN_INFO, config.TRAIN, model.training_transforms)
test_ds = get_dataset(config.TEST_INFO, config.TEST, model.validation_transforms)

test_dl = DataLoader(
    test_ds,
    batch_size=model.batch_size,
    shuffle=True
    )
print("[INFO] Dataset loaded succesfully")

##################################################################
#                    Model training
##################################################################
print("\n[INFO] Training model")
try:
    train_model(model, training_ds, test_dl, tb_writer)
except KeyboardInterrupt:
    print("Stopping training...")
print("[INFO] Training finished")

##################################################################
#                    Model evaluation
##################################################################
print("\n[INFO] Evaluating model")
acc = 0.0
try:
    acc = evaluate_model(model, test_dl)*100
    print(f'Accuracy: {acc:.5f} %')
except KeyboardInterrupt:
    print("Stopping evaluation...")
print("[INFO] Evaluation finished")

##################################################################
#               Save model + metadata
##################################################################
print("\n[INFO] Saving trained model metadata")
model_metadata.set_accuracy(acc)
torch.save(model.network.state_dict(), 
    os.path.join(trained_model_path, f"{model.id}.pth"))

# Add metadata to model info file
if not os.path.exists(config.MODELS_INFO_FILE_PATH):
    with open(config.MODELS_INFO_FILE_PATH, "w") as info_file:
        header = ",".join(ModelMetadata.serialization_attributes)
        info_file.write(f"{header}\n")
        info_file.write(str(model_metadata))
else:
    with open(config.MODELS_INFO_FILE_PATH, "a") as info_file:
        info_file.write(f"\n{model_metadata}")

print("[INFO] Saved successfully")