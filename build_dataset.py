#%% Imports
import os
from imutils import paths
import pandas as pd
import shutil
import sys
import numpy as np
from pandas.core.frame import DataFrame

import config

#%% Helper functions
def get_ds_size(max):
    ds_size = input(f"Dataset size (Max: {max}): ") or max

    return min(int(ds_size), max)

def get_test_split():
    split = input("Test split (def: 0.1): ") or 0.1

    return float(split)

def get_image_name(path):
    return os.path.split(path)[-1]

def get_label_from_image_path(path):
    return int(get_image_name(path)[0])

def copy_images(df, dest):
    # check if the destination folder exists and if not create it
    if os.path.exists(dest):
        clean = input(f"{dest} already exists, would you like to erase existing images before copying new images? [Y|N]: ")
        
        if clean.upper() == 'Y':
            shutil.rmtree(dest)
        else:
            print("Exiting to avoid mixing dataset")
            sys.exit()

    if not os.path.exists(dest):
        os.makedirs(dest)

    for row in df.values:
        origin = os.path.join(config.SOURCE_PATH, row[0])
        image_dest = os.path.join(dest, row[0])

        shutil.copy(origin, image_dest)

#%% Configure output directories
train_path = os.path.join(config.DATA_PATH, "training_data")
test_path = os.path.join(config.DATA_PATH, "test_data")

train_info = os.path.join(config.DATA_PATH, "train.csv")
test_info = os.path.join(config.DATA_PATH, "test.csv")

ds_info = os.path.join(config.DATA_PATH, "info.txt")

#%% Determine source type
source_type = input("Is the source files from sample(S) or origin(O) dataset? [S/O]: ").upper()

if not source_type in ["S","O"]:
    print("Invalid source type, valid types: S, O")
    sys.exit()

if source_type == "O":
    info_file = input("Specify path to info file: ")
    
    if not os.path.exists(info_file): 
        print("Info file doesn't exist")
        sys.exit()

#%% Get get image data
df: DataFrame = None
ds_size: int = 0

if source_type == "O":
    data_info = pd.read_csv(info_file)
    df = data_info.sample(frac=1).reset_index(drop=True)
    ds_size = get_ds_size(len(df))
    df = df.loc[:ds_size - 1]
else:
    image_paths = list(paths.list_images(config.SOURCE_PATH))
    ds_size = get_ds_size(len(image_paths))
    image_paths = image_paths[:ds_size]
    df = pd.DataFrame(image_paths, columns=["path"])
    

#%% Create dataframe with type
if source_type == "O":
    df["name"] = df["img_dir"].apply(get_image_name)
    df["type"] = df["img_labels"]
    df = df.drop(["index","img_dir","img_labels"],axis=1)
else:
    df["name"] = df["path"].apply(get_image_name)
    df["type"] = df["path"].apply(get_label_from_image_path)
    df = df.drop(["path"],axis=1)

#%% Split into test and training sets
test_split = get_test_split()
test_len = int(len(df) * test_split)
test_df = df.loc[:test_len-1]
train_df = df.loc[test_len:]

#%% Copy images to respective folders
copy_images(test_df, test_path)
copy_images(train_df, train_path)

#%% Create information files
test_df.to_csv(test_info)
train_df.to_csv(train_info)

#%% Dataset info 
print("Dataset created")

train_len = len(train_df)
n_pos = train_df["type"].sum()
n_neg = train_len - n_pos

info_text = f"""
Dataset stats:
    Size: {ds_size}
    Training size: {len(train_df)}
    Test size: {test_len}

    Training weight info:
        Amount of negative samples: {n_neg}
        Amount of positive samples: {n_pos}
        pos_weight: {n_neg/n_pos}
"""

print(info_text)

with open(ds_info, "w") as f:
    f.write(info_text)