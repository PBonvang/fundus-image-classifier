#%% Imports
import os
from imutils import paths
import pandas as pd
import shutil
import numpy as np

import config

DATA_INFO = pd.read_csv("M:\\Datasets\\IsFundus 2022-01-05 13-42\\train.csv")

#%% Helper functions
def get_image_name(path):
    return os.path.split(path)[-1]

def get_label_from_image_path(path):
    return int(get_image_name(path)[0])

def copy_images(df, dest):
    # check if the destination folder exists and if not create it
    if os.path.exists(dest):
        clean = input(f"{dest} already exists, would you like to erase existing images before copying new images? [Y|N]:")
        
        if clean.upper() == 'Y':
            shutil.rmtree(dest)

    if not os.path.exists(dest):
        os.makedirs(dest)

    for row in df.values:
        path = os.path.join(config.SOURCE_PATH, row[0])
        image_dest = os.path.join(dest, row[0])

        shutil.copy(path, image_dest)

#%% Configure output directories
train_path = os.path.join(config.DATA_PATH, "training_data")
test_path = os.path.join(config.DATA_PATH, "test_data")

train_info = os.path.join(config.DATA_PATH, "train.csv")
test_info = os.path.join(config.DATA_PATH, "test.csv")

#%% Get paths
df = DATA_INFO.sample(frac=1).reset_index(drop=True)
ds_size = 5000
df = df.loc[:ds_size - 1]

#%% Create dataframe with type
df["name"] = df["img_dir"].apply(get_image_name)
df["type"] = df["img_labels"]
df = df.drop(["index","img_dir","img_labels"],axis=1)

#%% Split into test and training sets
test_len = int(len(df) * 0.1)
test_df = df.loc[:test_len-1]
train_df = df.loc[test_len:]

#%% Copy images to respective folders
copy_images(test_df, test_path)
copy_images(train_df, train_path)

#%% Create information files
test_df.to_csv(test_info)
train_df.to_csv(train_info)

#%% Dataset info 
print("Clean dataset created")

train_len = len(train_df)
n_pos = train_df["type"].sum()
n_neg = train_len - n_pos

print(f"""
Training weight info:
    Amount of negative samples: {n_neg}
    Amount of positive samples: {n_pos}
    pos_weight: {n_neg/n_pos}
""")