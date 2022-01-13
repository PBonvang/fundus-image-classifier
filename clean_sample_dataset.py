#%% Imports
import os
from imutils import paths
import pandas as pd
import shutil

import config

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
        path = row[0]
        if path == "nan":
            continue
        
        image_dest = os.path.join(dest, row[1])

        shutil.copy(path, image_dest)

#%% Configure output directories
train_path = os.path.join(config.DATA_PATH, "training_data")
test_path = os.path.join(config.DATA_PATH, "test_data")

train_info = os.path.join(config.BASE_PATH, "train.csv")
test_info = os.path.join(config.BASE_PATH, "test.csv")

#%% Get paths
image_paths = list(paths.list_images(config.SOURCE_PATH))
ds_size = len(image_paths)
image_paths = image_paths[:ds_size]

#%% Create dataframe with type
df = pd.DataFrame(image_paths, columns=["path"])
df["name"] = df["path"].apply(get_image_name)
df["type"] = df["path"].apply(get_label_from_image_path)

#%% Split into test and training sets
test_len = int(len(df) * 0.1)
test_df = df.loc[:test_len]
train_df = df.loc[test_len:]

#%% Copy images to respective folders
copy_images(test_df, test_path)
copy_images(train_df, train_path)

#%% Create information files
test_df = test_df.drop("path", axis=1)
train_df = train_df.drop("path", axis=1)

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