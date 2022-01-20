import os
import pandas as pd

source = r"M:\\Datasets\\IsFundus 2022-01-05 13-42\\validation.csv"
dest = "M:\\Datasets\\IsFundus-test-set\\test.csv"

df = pd.read_csv(source)
df["name"] = df["img_dir"].apply(lambda n: os.path.split(n)[-1])
df["type"] = df["img_labels"]
df = df.drop(["index","img_dir","img_labels"],axis=1)

df.to_csv(dest)