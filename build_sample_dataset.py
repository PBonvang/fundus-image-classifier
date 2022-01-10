from imutils import paths
import config
from utils.SampleType import SAMPLE_TYPE

import numpy as np
import shutil
import os

def copy_images(image_paths, dest):
    # check if the destination folder exists and if not create it
    if os.path.exists(dest):
        clean = input("Destination folder already exists, would you like to erase existing images it before copying new images? [Y|N]:")
        
        if clean.upper() == 'Y':
            shutil.rmtree(dest)

    if not os.path.exists(dest):
        os.makedirs(dest)

    classes = ['not_fundus','fundus']
    for sub_path in classes:
        full_sub_path = os.path.join(dest,sub_path)

        if not os.path.exists(full_sub_path):
            os.makedirs(full_sub_path)

    i = 1
    for image_path in image_paths:
        if image_path == 'nan':
            continue

        image_name = image_path.split(os.path.sep)[-1]
        name, ext = os.path.splitext(image_name)
        image_type = SAMPLE_TYPE.to_sample_type(image_name)
        image_dest = os.path.join(dest, classes[image_type.value], f"{name}-{i}.{ext}")

        shutil.copy(image_path, image_dest)
        
        i += 1

print("[INFO] loading training data...")
image_paths = list(paths.list_images(config.SOURCE_PATH))
np.random.shuffle(image_paths)

val_len = int(len(image_paths) * config.VAL_SPLIT)
train_len = len(image_paths) - val_len

training_paths = image_paths[:train_len]
validation_paths = image_paths[train_len:]

print("[INFO] copying training and test images...")
copy_images(training_paths, config.TRAIN)
copy_images(validation_paths, config.VAL)