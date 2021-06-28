# %%
import xml.etree.ElementTree as ET
import glob
import pathlib
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
input_path = pathlib.Path('./PascalVOCFaceMasks')
out_path = input_path / "subset"
list_xmls = list(input_path.rglob('./**/*.xml'))
list_pngs = list(input_path.rglob('./**/*.png'))
zipped_list = {xmlf.name: [xmlf, imgf] for xmlf, imgf in zip(list_xmls, list_pngs)}
# %%
label_mapping = {
    "with_mask": 1,
    # "mask_weared_incorrect": -1,
    "without_mask": 0
}


def extract_obj_inf(fname, xml_obj, key, img_path):
    label = label_mapping.get(xml_obj.findtext("./name"), -1)
    bbox = xml_obj.find("./bndbox")
    xmin = int(bbox.findtext("xmin"))
    ymin = int(bbox.findtext("ymin"))
    xmax = int(bbox.findtext("xmax"))
    ymax = int(bbox.findtext("ymax"))
    return {
        "file": fname,
        "Mask_on": label,
        "x1": xmin,
        "x2": xmax,
        "y1": ymin,
        "y2": ymax,
        "key":key,
        "img_path":img_path.absolute(),
    }


xml_roots = [(key, ET.parse(xml_path).getroot(), img_path) for key, (xml_path, img_path) in zipped_list.items()]
xml_data = [(xml_obj.findtext("./filename"), xml_obj.findall("./object")[0], key, img_path)
            for key, xml_obj, img_path in xml_roots if len(xml_obj.findall("./object")) == 1]
flattened_xml_data = [extract_obj_inf(*obj) for obj in xml_data]
flattened_xml_data
# %%
data = pd.DataFrame(flattened_xml_data)
data = data[data.Mask_on != -1]
data.to_csv("./PascalVOCFaceMasks/pascal_voc_dataset.csv")
data
# %%
for category, grp in data.groupby(["Mask_on"]):
    print(f"{category} has {len(grp)} images")
    for idx, series in grp.iterrows():
        start = series.get("img_path") 
        img_name = start.name
        parent = start.parent

        destination = out_path / ("mask_on" if category else "mask_off") / img_name
        a = start.absolute()
        b = destination.absolute()
        if not destination.parent.exists():
            print(f"{a} -> {b}")
            os.makedirs(destination.parent.absolute())
        shutil.copy(a, b)