from imutils import paths
import config
from utils.SampleType import SAMPLE_TYPE

import numpy as np
import shutil
import os

def get_evenly_weigthed_classes_paths(image_paths):
    fundus_paths = list()
    non_fundus_paths = list()
    for path in image_paths:
        if path == 'nan':
            continue

        image_name = path.split(os.path.sep)[-1]
        image_type = SAMPLE_TYPE.to_sample_type(image_name)
        if image_type == SAMPLE_TYPE.FUNDUS:
            fundus_paths.append(path)
        else:
            non_fundus_paths.append(path)

    m = int(np.ceil(len(fundus_paths)/len(non_fundus_paths)))
    non_fundus_paths = (non_fundus_paths*m)[:len(fundus_paths)]
    evenly_weighted_paths = fundus_paths + non_fundus_paths

    return evenly_weighted_paths

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
evenly_weighted_paths = get_evenly_weigthed_classes_paths(image_paths)
np.random.shuffle(evenly_weighted_paths)

val_len = int(len(evenly_weighted_paths) * config.VAL_SPLIT)
train_len = len(evenly_weighted_paths) - val_len

training_paths = evenly_weighted_paths[:train_len]
validation_paths = evenly_weighted_paths[train_len:]

print("[INFO] copying training and test images...")
copy_images(training_paths, config.TRAIN)
copy_images(validation_paths, config.VAL)