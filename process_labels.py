# %%
import json
import requests
import os
import shutil
from urllib import request
import jsonlines
import io
import pandas as pd
import pathlib
from tqdm import tqdm
# %%
# https://www.kaggle.com/dataturks/face-detection-in-images
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

root = pathlib.Path('./label - Kopie')
input_path = root / "imgs"
annot_path = root / "annotations"
# reader = jsonlines.Reader(fp)
all_files = {f.name: f for pattern in ['./**/*.png', './**/*.jp*g'] for f in input_path.rglob(pattern)}
all_annotations = {f.name: f for pattern in ['./**/*.xml'] for f in input_path.rglob(pattern)}
# %%
all_files_labelled = []
for key, file in all_files.items():
    pre_dirs = [
        file.parent.parent.parent.parent,
        file.parent.parent.parent,
        file.parent.parent,
        file.parent,
    ]
    info = all_annotations.get(key, None)
    if info is None:
        print(bcolors.WARNING + f"Warning: {key} is none!" + bcolors.ENDC)
    lbl, race, sex, skinc = [pth.name for pth in pre_dirs]
    lbl_file = {"Mask_on": lbl, "Race": race, "Sex": sex, "SkinColor": skinc, "info": info}
    all_files_labelled.append(lbl_file)
# %%
data = pd.DataFrame(all_files)
data
# %%
for idx, series in tqdm(data.iterrows()):
    p = series.get("path")
    online_location = series.get("path")
    fname = p.name
    outfile = outpath.absolute() / fname
    print(str(online_location), str(outfile))
    request.urlretrieve(str(online_location), str(outfile))

# %%
label_mapping = {
    "with_mask": 1,
    # "mask_weared_incorrect": -1,
    "face": 0
}
race_mapping = {
    "indian": 4,
    "asian": 3,
    "black": 2,
    "other": 1,
    "caucasian": 0,
}
sex_mapping = {"male": 1, "female": 0}
