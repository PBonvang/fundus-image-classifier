import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import enum
import pandas as pd
import time

BASE_PATH = '/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage'
N_EPOCHS = 10
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SAMPLE_TYPE(enum.Enum):
    NOT_FUNDUS = 0
    FUNDUS = 1

    def to_sample_type(hot_repr: str):
        if hot_repr[0] == '1':
            return SAMPLE_TYPE.FUNDUS

        return SAMPLE_TYPE.NOT_FUNDUS

def load_data(info_file="Train.txt"):
    training_info = pd.read_csv(f"{BASE_PATH}/{info_file}", sep='\t', lineterminator='\r',
    header=None,
    names=[
        'idx',
        'image_name',
        'type'
    ],
    nrows=1024) # Makes my computer crash if full list is used, due to limited ram (8 GB)

    training_info["image_name"] = training_info["image_name"].astype(str)
    training_info["image_name"] = training_info["image_name"].apply(lambda fp: fp.replace("S:\\CNTKFiles\\IsFundusImage\\images\\",""))
    training_info["type"] = training_info["type"].apply(lambda t: SAMPLE_TYPE.to_sample_type(str(t)))
    training_info["image"] = training_info.apply(lambda row: read_image_file(f"{BASE_PATH}/Images/{row.image_name}"), axis=1)

    training_data = torch.tensor(training_info['image'].values)
    training_target = torch.tensor(training_info['type'].values.astype(np.float32))
    training_dataset = data_utils.TensorDataset(training_data, training_target)
    training_loader = data_utils.DataLoader(training_dataset, batch_size= 32, shuffle=True)

    return training_loader

def read_image_file(file_path):
    x = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    #x = cv2.resize(x,(256,256))

    return x

start_time = time.perf_counter()

train_dl = load_data()

end_time = time.perf_counter()
print(f"Execution time: {end_time-start_time}s")