import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import enum
import pandas as pd
import time

DATA_PATH = '/home/master/Documents/Study/IntroToIntelligentSystems/FinalProject/StatuManu/IsFundusImage'
N_EPOCHS = 10
BATCH_SIZE = 64

class SAMPLE_TYPE(enum.Enum):
    NOT_FUNDUS = 0
    FUNDUS = 1

    def to_sample_type(hot_repr: str):
        if hot_repr[0] == '1':
            return SAMPLE_TYPE.FUNDUS

        return SAMPLE_TYPE.NOT_FUNDUS

def load_data(info_file="Train.txt"):
    training_info = pd.read_csv(f"{DATA_PATH}/{info_file}", sep='\t', lineterminator='\r',
    header=None,
    names=[
        'idx',
        'filepath',
        'type'
    ],
    nrows=512) # Makes my computer crash if full list is used, because of limited ram (8 GB)

    training_info["filepath"] = training_info["filepath"].astype(str)
    training_info["filepath"] = training_info["filepath"].apply(lambda fp: fp.replace("S:\\CNTKFiles\\IsFundusImage\\images\\",""))
    training_info["type"] = training_info["type"].apply(lambda t: SAMPLE_TYPE.to_sample_type(str(t)))
    training_info["image"] = training_info.apply(lambda row: read_image_file(f"{DATA_PATH}/Images/{row.filepath}"), axis=1)

    return [], []

def read_image_file(file_path):
    x = cv2.imread(file_path)
    #x = cv2.resize(x,(256,256))

    return x

start_time = time.perf_counter()

training_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

end_time = time.perf_counter()
print(f"Execution time: {end_time-start_time}s")