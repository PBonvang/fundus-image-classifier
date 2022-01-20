import torch
import os

#############################################
# Debugging configuration
#############################################
DEBUG = False # Debugging switch, enabling and disabling debugging logs

#############################################
# Dataset configuration
#############################################
SOURCE_PATH = "Path/to/fundus_images" # Path to folder containing fundus images
DATA_PATH = "Path/to/dataset" # Path to dataset used for training and validation



TRAIN = os.path.join(DATA_PATH, "training_data") # Path to training dataset
TRAIN_INFO = os.path.join(DATA_PATH, "train.csv") # Path to training label file
DS_WEIGHT = 0.29158215010141986 # Dataset positive weight, to counterbalance imbalanced datasets

TEST = os.path.join(DATA_PATH, "test_data") # Path to test/validation dataset
TEST_INFO = os.path.join(DATA_PATH, "test.csv") # Path to test/validation label file

IMAGE_SHAPE = (224, 179)

#############################################
# Model configuration
#############################################
TRAINED_MODELS_PATH = "models" # Path to store model training runs in
MODEL_DEF = "Model.py" # Path to the python file containing the model blueprint 
MODELS_INFO_FILE = os.path.join(TRAINED_MODELS_PATH, "models_info.csv") # CSV file to save model info to

SAVE_EPOCH_CHECKPOINTS = False # Enabled a checkpoint will be saved after each epoch
SAVE_STEP_CHECKPOINTS = False # Enabled a checkpoint will be saved every tenth of an epoch

#############################################
# Hypermodel configuration
#############################################
HYPER_MODEL_DEF = "HyperModel.py" # Path to the python file containing the hypermodel blueprint
HYPER_MODELS_PATH = "hypermodels" # Path to store hypermodel blueprints
STUDIES_PATH = "studies" # Path to save optuna studies to

#############################################
# CUDA configuration
#############################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
