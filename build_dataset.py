from imutils import paths
import config
from Utils import SAMPLE_TYPE

import numpy as np
import shutil
import os
import pandas as pd

def copy_images(image_paths, dest):
    # check if the destination folder exists and if not create it
    if os.path.exists(dest):
        clean = input("Destination folder already exists, would you like to erase existing images it before copying new images? [Y|N]:")
        
        if clean.upper() == 'Y' or clean == "":
            shutil.rmtree(dest)

    if not os.path.exists(dest):
        os.makedirs(dest)

    classes = ['not_fundus','fundus']
    for sub_path in classes:
        full_sub_path = os.path.join(dest,sub_path)

        if not os.path.exists(full_sub_path):
            os.makedirs(full_sub_path)

    for image_path in image_paths:
        if image_path == 'nan':
            continue

        image_name = image_path.split(os.path.sep)[-1]
        image_type = SAMPLE_TYPE.to_sample_type(image_name)
        image_dest = os.path.join(dest, classes[image_type.value], image_name)
        if not os.path.exists(image_dest):
            shutil.copy(image_path, image_dest)

print("[INFO] loading training data...")
image_paths = list(paths.list_images(config.SOURCE_PATH))\
    [:256]# Else the program is killed, due to limited memory (8 GB)
np.random.shuffle(image_paths)

val_len = int(len(image_paths) * config.VAL_SPLIT)
train_len = len(image_paths) - val_len

training_paths = image_paths[:train_len]
validation_paths = image_paths[train_len:]

print("[INFO] copying training and test images...")
copy_images(training_paths, config.TRAIN)
copy_images(validation_paths, config.VAL)