import torch
import os

# define path to the original dataset and base path to the dataset
SOURCE_PATH = "/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage/Images"
DATA_PATH = os.path.join("/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage/", "Data")

MODELS_PATH = "final_models"
TRAINED_MODELS_PATH = "trained_models"
MODEL_DEF = "Model.py"
MODELS_INFO_FILE = os.path.join(MODELS_PATH, "model_info.csv")

HYPER_MODEL_DEF = "HyperModel.py"
HYPER_MODELS_PATH = "hyper_models"

# define paths to separate train and test
TRAIN = os.path.join(DATA_PATH, "training_data")
TRAIN_INFO = os.path.join(DATA_PATH, "train.csv")
DS_WEIGHT = 0.29158215010141986

TEST = os.path.join(DATA_PATH, "test_data")
TEST_INFO = os.path.join(DATA_PATH, "test.csv")

IMAGE_SHAPE = (256, 256)

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# define study config
STUDIES_PATH = "./studies"

DEBUG = True