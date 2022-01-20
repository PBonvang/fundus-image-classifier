#%% imports
import shutil
import uuid
import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from Model import get_model
import utils
from utils import RunInfo, IModel
import config

##################################################################
#%%                    Model initialization
##################################################################
print("[INFO] Initializing model")
model = get_model()

if not utils.model_is_valid(model):
    raise TypeError(
        f"Model is missing attributes. Please define missing attributes and try again. Required attributes are defined in the {IModel.__name__} interface.")

load_existing = input(
    "Do you wish to create new weights(N) or load existing(E)? ").upper() == "E"
base_model_path = None
if load_existing:
    base_model_path = input("Input path to model state dict file: ")

    if not os.path.exists(base_model_path):
        print(f"Couldn't find model at {base_model_path}")
        sys.exit()

    if not os.path.splitext(base_model_path)[1] in [".pth"]:
        print(f"Invalid file extension found, only takes .pth")
        sys.exit()

    model_id: str = None
    try:
        model_id = base_model_path.split(config.TRAINED_MODELS_PATH)[1]\
            .split(os.sep)[1]

        model_id_input = input(f"Input model id ({model_id}): ")
        if model_id_input != "":
            model_id = model_id_input
    except:
        model_id = str(uuid.uuid4())

    model.id = model_id

    model.network.load_state_dict(torch.load(base_model_path))

print("Model id: ", model.id)
print("[INFO] Model initialized")

##################################################################
#%%                    Prepare run
##################################################################
print("\n[INFO] Preparing run")
trained_model_path = os.path.join(config.TRAINED_MODELS_PATH, model.id)
run_id = str(uuid.uuid4())

print(f"Run id: ", run_id)

run_path = os.path.join(trained_model_path, "runs", run_id)

print("\n[INFO] Loading dataset")
training_ds = utils.get_dataset(
    config.TRAIN_INFO, config.TRAIN, model.training_transforms)
test_ds = utils.get_dataset(config.TEST_INFO, config.TEST,
                      model.validation_transforms)

test_dl = DataLoader(
    test_ds,
    batch_size=model.batch_size,
    shuffle=True
)
print("[INFO] Dataset loaded succesfully")


run_info = RunInfo(
    id=run_id,
    name=input(f"Specify run name [Optional]: "),
    model=model,
    run_path=run_path,
    ds_size=len(training_ds),
    base_model_path=base_model_path
)

os.makedirs(run_info.checkpoint_path)

print("\n[INFO] Saving model blueprint")
shutil.copy(config.MODEL_DEF,
            os.path.join(run_path, f"{model.id}.py"))
print("[INFO] Model blueprint saved")

tb_writer = SummaryWriter(
    os.path.join(run_path, "tensorboard")
)
print("[INFO] Run prepared")

##################################################################
#%%                    Train model
##################################################################
print("\n[INFO] Training model")
try:
    utils.train_model(model,
                training_ds,
                test_dl,
                run_info,
                tb_writer)
except KeyboardInterrupt:
    print("Stopping training...")
print("[INFO] Training finished")

##################################################################
#%%                    Model evaluation
##################################################################
print("\n[INFO] Evaluating model")
acc = 0.0
try:
    acc = utils.evaluate_model(model, test_dl, run_path)*100
    print(f'Accuracy: {acc:.5f} %')
except KeyboardInterrupt:
    print("Stopping evaluation...")
print("[INFO] Evaluation finished")

run_info.set_test_accuracy(acc)

##################################################################
#%%               Save model + run info
##################################################################
print("\n[INFO] Saving trained model and run info")
torch.save(model.network.state_dict(),
           os.path.join(run_info.run_path, f"{run_info.id}.pth"))

run_info.save_to_csv(config.MODELS_INFO_FILE)

print("[INFO] Saved successfully")