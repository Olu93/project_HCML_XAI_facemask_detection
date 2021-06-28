# %%
import xml.etree.ElementTree as ET
import glob
import pathlib
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# %%
input_path = pathlib.Path('./faces_only/subset')
out_path = pathlib.Path('./faces_only/to-label/')
list_xmls = {path.stem: path for path in list(input_path.rglob('./**/*.xml'))}
list_pngs = {path.stem: path for path in list(input_path.rglob('./**/*.png'))}
list_jpgs = {path.stem: path for path in list(input_path.rglob('./**/*.jp*g'))}
list_imgs = dict(**list_pngs, **list_jpgs)

zipped_list = {key: {"xml": xmlf, "img": list_imgs.get(key)} for key, xmlf in list_xmls.items()}
pprint(zipped_list)
# %%
label_mapping = {
    "with_mask": 1,
    # "mask_weared_incorrect": -1,
    "face": 0
}

race_mapping = {
    "indian":4,
    "asian":3,
    "black":2,
    "other":1,
    "caucasian":0,
}

sex_mapping = {
    "male":1,
    "female":0
}

def extract_obj_inf(fname, xml_obj, key, img_path):
    label = label_mapping.get(xml_obj.findtext("./name"), -1)
    race, sex = img_path.parent.parent.name, img_path.parent.name
    bbox = xml_obj.find("./bndbox")
    xmin = int(bbox.findtext("xmin"))
    ymin = int(bbox.findtext("ymin"))
    xmax = int(bbox.findtext("xmax"))
    ymax = int(bbox.findtext("ymax"))
    return {
        "Img": fname,
        "Person_num": 1,
        "Mask_on": label,
        "Race": race_mapping.get(race),
        "SkinColor": "",
        "Sex": sex_mapping.get(sex),
        "x1": xmin,
        "x2": xmax,
        "y1": ymin,
        "y2": ymax,
        "key": key,
        "img_path": img_path.absolute(),
    }


xml_roots = [(key, ET.parse(files.get("xml")).getroot(), files.get("img")) for key, files in zipped_list.items()]
xml_data = [(xml_obj.findtext("./filename"), xml_obj.findall("./object")[0], key, img_path)
            for key, xml_obj, img_path in xml_roots if len(xml_obj.findall("./object")) == 1]
flattened_xml_data = [extract_obj_inf(*obj) for obj in xml_data]
pprint(flattened_xml_data)
# %%
data = pd.DataFrame(flattened_xml_data)
data = data[data.Mask_on != -1].sort_values("Img")
data
# %%
data.to_csv("./faces_only/data.csv")
# %%
# filenames = np.sort(data.Img.values())
for idx, series in data.iterrows():
    start = series.get("img_path")
    img_name = series.get("Img")
    destination = out_path / img_name
    a = start.absolute()
    b = destination.absolute()
    if not destination.parent.exists():
        print(f"{a} -> {b}")
        os.makedirs(destination.parent.absolute())
    shutil.copy(a, b)
# for category, grp in data.groupby(["Mask_on"]):
#     print(f"{category} has {len(grp)} images")
#     for idx, series in grp.iterrows():
#         start = series.get("img_path")
#         img_name = start.name
#         parent = start.parent

#         destination = out_path / ("mask_on" if category else "mask_off") / img_name
#         a = start.absolute()
#         b = destination.absolute()
#         if not destination.parent.exists():
#             print(f"{a} -> {b}")
#             os.makedirs(destination.parent.absolute())
#         shutil.copy(a, b)