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
fp = io.open('face_detection.json')  # readable file-like object
outpath = pathlib.Path("./faces_only/imgs")
reader = jsonlines.Reader(fp)
all_files = []
for obj in reader:
    fname = obj["content"]
    annotations = obj["annotation"]
    if len(annotations) > 1:
        continue
    ann = annotations[0]
    label = ann["label"][0]
    startp, endp = ann["points"]
    xmin = float(startp.get("x"))
    ymin = float(startp.get("y"))
    xmax = float(endp.get("x"))
    ymax = float(endp.get("y"))
    file_type = pathlib.Path(fname).suffix
    all_files.append({
        "file": fname,
        "path": pathlib.Path(fname),
        "Mask_on": label,
        "x1": xmin,
        "x2": xmax,
        "y1": ymin,
        "y2": ymax,
        "file_type": file_type,
    })
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
