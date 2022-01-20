"""
This script gathers all samples in the provided dataset and copies them to a folder including a info file containing the image labels.
"""
#%% Imports
import os
from os.path import exists
import pandas as pd
import shutil
import sys

import config

#%% Helper functions
def get_image_name(path):
    return os.path.split(path)[-1]

def copy_images(df, source, dest):
    # check if the destination folder exists and if not create it
    if exists(dest):
        clean = input(f"{dest} already exists, would you like to erase existing images before copying new images? [Y|N]: ")
        
        if clean.upper() == 'Y':
            shutil.rmtree(dest)
        else:
            print("Exiting to avoid mixing dataset")
            sys.exit()

    if not exists(dest):
        os.makedirs(dest)

    for row in df.values:
        split = os.path.split(row[0]) 
        type = split[0][1:]
        name = split[1]
        origin = os.path.join(source, type, name)
        image_dest = os.path.join(dest, name)

        shutil.copy(origin, image_dest)

#%% Configure output directories
dest_path = input(f"Specify output directory ({config.DATA_PATH}): ") or config.DATA_PATH
if not exists(dest_path):
    create_dest = input(f"The directory {dest_path} doesn't exist, do you wish to create it? [Y/N]").upper()

    if create_dest == "Y":
        os.makedirs(dest_path)
    else:
        sys.exit()

test_path = os.path.join(dest_path, "test_data")
test_info = os.path.join(dest_path, "test.csv")

ds_info = os.path.join(dest_path, "info.txt")

#%% Determine source
source_path = input(f"Specify source directory ({config.SOURCE_PATH}): ") or config.SOURCE_PATH
if not exists(source_path):
    print(f"Invalid path specified: {source_path}")
    sys.exit()

train_info_file = os.path.join(source_path, "train.csv")
val_info_file = os.path.join(source_path, "validation.csv")

if not exists(train_info_file) \
    or not exists(val_info_file)\
    or not exists(os.path.join(source_path,"train"))\
    or not exists(os.path.join(source_path,"validation")):
    print("Specified source is missing some of the missing elements: train.csv, validation.csv, train/, validation/")
    sys.exit()

#%% Get get image labels
train_df = pd.read_csv(train_info_file)
val_df = pd.read_csv(val_info_file)
df = train_df.merge(val_df, how="outer")

#%% Create dataframe with type
df["name"] = df["img_dir"].apply(get_image_name)
df["type"] = df["img_labels"]
df = df.drop(["index","img_labels"],axis=1)

#%% Copy images to respective folders
copy_images(df, source_path, test_path)

#%% Create information files
df = df.drop(["img_dir"],axis=1)
df.to_csv(test_info)

#%% Dataset info 
print("Dataset created")

info_text = f"""
Dataset stats:
    Size: {len(df)}
"""

with open(ds_info, "w") as f:
    f.write(info_text)

print(info_text)