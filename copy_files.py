# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
import shutil
import pathlib
# %%
race_dir = pathlib.Path('./classes/race')
race_sex_dir = pathlib.Path('./classes/race_sex')
skin_sex_dir = pathlib.Path('./classes/skin_sex')
dataset_dir = pathlib.Path('./Medical mask/Medical mask/images')
biased_dataset_labels = pd.read_csv('./dataset.csv')
biased_dataset_labels

# %%
df = biased_dataset_labels.loc[biased_dataset_labels.Is_Complicated == False]
df = df.loc[biased_dataset_labels.Race != 1]
# %%
for category, grp in df.groupby(["Mask_on", "Race"]):
    list_of_images = list(grp.Img)
    has_mask, lbl = category
    print(f"{category[0]} - {category[1]} has {len(list_of_images)} images")
    for img_name in list_of_images:
        start = dataset_dir / img_name
        destination = race_dir / ("mask_on" if has_mask else "mask_off") / str(lbl) / img_name
        a = start.absolute()
        b = destination.absolute()
        if not destination.parent.exists():
            print(f"{a} -> {b}")
            os.makedirs(destination.parent.absolute())
        shutil.copy(a, b)
# %%
for category, grp in df.groupby(["Mask_on", "Race", "Sex"]):
    list_of_images = list(grp.Img)
    has_mask, race_lbl, sex_lbl = category
    print(f"{category[0]} - {category[1]} has {len(list_of_images)} images")
    for img_name in list_of_images:
        sub_path = f"{str(race_lbl)}-{str(sex_lbl)}"
        start = dataset_dir / img_name

        destination = race_sex_dir / ("mask_on" if has_mask else "mask_off") / sub_path / img_name
        a = start.absolute()
        b = destination.absolute()
        if not destination.parent.exists():
            print(f"{a} -> {b}")
            os.makedirs(destination.parent.absolute())
        shutil.copy(a, b)
# %%
for category, grp in df.groupby(["Mask_on", "SkinColor", "Sex"]):
    list_of_images = list(grp.Img)
    has_mask, skin_lbl, sex_lbl = category
    print(f"{category[0]} - {category[1]} has {len(list_of_images)} images")
    for img_name in list_of_images:
        sub_path = f"{str(skin_lbl)}-{str(sex_lbl)}"
        start = dataset_dir / img_name

        destination = skin_sex_dir / ("mask_on" if has_mask else "mask_off") / sub_path / img_name
        a = start.absolute()
        b = destination.absolute()
        if not destination.parent.exists():
            print(f"{a} -> {b}")
            os.makedirs(destination.parent.absolute())
        shutil.copy(a, b)
# %%
import glob
filesets = [list(dataset_dir.glob(f"*.{ext}")) for ext in "png jpg jpeg".split()]
all_files = []
# all_files.extend()
for fset in filesets:
    all_files.extend([f.name for f in fset])

with open("tmp.txt", "w") as tmpf:
    for fl in all_files:
        tmpf.write(fl + "\n")
# %%
# %%
median_counts = df.groupby(["Mask_on", "Race", "Sex"])["Img"].count().reset_index().groupby("Mask_on").Img.median()
# %%
df.groupby(["Mask_on", "Race", "Sex"]).count().reset_index().Img.median()
# %%
