import torch
import os
import datetime

# define path to the original dataset and base path to the dataset
BASE_PATH = "/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage"
DATA_PATH = os.path.join(BASE_PATH, "Data")
SOURCE_PATH = os.path.join(BASE_PATH, "Images")
MODELS_PATH = "./models"
TRAINED_MODELS_PATH = "./trained_models"
MODEL_DEF = "./Model.py"
MODELS_INFO_FILE_PATH = os.path.join(MODELS_PATH, "model_info.csv")
#SAVE_PATH = os.path.join(MODEL_PATH, f"{datetime.datetime.now():%d_%m_%Y__%H_%M_%S}.pth")

# define paths to separate train and test
VAL_SPLIT = 0.1
TRAIN = os.path.join(DATA_PATH, "train")
VAL = os.path.join(DATA_PATH, "validation")

IMAGE_SIZE = 256

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify training hyperparameters
FEATURE_EXTRACTION_BATCH_SIZE = 16
FINETUNE_BATCH_SIZE = 4
PRED_BATCH_SIZE = 4
EPOCHS = 2
LR = 0.0001
LR_FINETUNE = 0.0005

# define paths to store training plots and trained model
WARMUP_PLOT = os.path.join(BASE_PATH, "Output", "warmup.png")
FINETUNE_PLOT = os.path.join(BASE_PATH, "Output", "finetune.png")
WARMUP_MODEL = os.path.join(BASE_PATH, "Output", "warmup_model.pth")
FINETUNE_MODEL = os.path.join(BASE_PATH, "Output", "finetune_model.pth")

###########################################################
#                   SUPER SETTINGS                        #
###########################################################
ON_SUPER_COM = False
TRAIN_INFO = os.path.join(BASE_PATH, "train.csv")
VAL_INFO = os.path.join(BASE_PATH, "validation.csv")