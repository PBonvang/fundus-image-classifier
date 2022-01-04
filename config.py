import torch
import os

# define path to the original dataset and base path to the dataset
BASE_PATH = "/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage"
DATA_PATH = os.path.join(BASE_PATH, "Data")

# define paths to separate train and test
TRAIN = os.path.join(BASE_PATH, "train")
TEST = os.path.join(BASE_PATH, "test")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify training hyperparameters
FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 64
PRED_BATCH_SIZE = 4
EPOCHS = 20
LR = 0.001
LR_FINETUNE = 0.0005

# define paths to store training plots and trained model
WARMUP_PLOT = os.path.join(BASE_PATH, "Output", "warmup.png")
FINETUNE_PLOT = os.path.join(BASE_PATH, "Output", "finetune.png")
WARMUP_MODEL = os.path.join(BASE_PATH, "Output", "warmup_model.pth")
FINETUNE_MODEL = os.path.join(BASE_PATH, "Output", "finetune_model.pth")