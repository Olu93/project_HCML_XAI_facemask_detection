# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
import shutil
import pathlib
import json
import xml
import io
from pprint import pprint
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# %%
patterns_img = ["png", "jp*g"]
patterns_info = ["xml", "json"]
base_initial_dir = pathlib.Path('./Medical mask/Medical mask/images')
base_custom_dir = pathlib.Path('./faces_only/subset')
base_info_initial_dir = pathlib.Path('./Medical mask/Medical mask/annotations')
base_info_custom_dir = pathlib.Path('./faces_only/subset')
unbiased_dataset = pd.read_csv('./csv_datasets/dataset.csv')
debiased_dataset = pd.read_csv('./csv_datasets/additional_dataset_2.csv')
# %%
base_dirs = [base_initial_dir, base_custom_dir]
# dataset = pd.concat([unbiased_dataset, debiased_dataset])
dataset = debiased_dataset.copy()
dataset = dataset[~pd.isnull(dataset.Mask_on)]
dataset = dataset.loc[~dataset.Is_Complicated.astype(bool)]
dataset = dataset.loc[dataset.Race != 1]
dataset = dataset.set_index('Img')
dataset
# %%
list_of_selected_images = list(dataset.index)
list_of_selected_images
# %%print
list_of_all_images_available = []
for extenstion in patterns_img:
    for base_dir in base_dirs:
        list_of_all_images_available.extend(
            (img for img in base_dir.rglob(f"*.{extenstion}") if img.name in list_of_selected_images))
with open("./all_available_imgs.txt", "w") as f:
    for pth in list_of_all_images_available:
        f.write(str(pth) + "\n")
name_path_mapping = {pth.name: pth for pth in list_of_all_images_available}
file_locations = pd.DataFrame(name_path_mapping.items(), columns="key file".split()).set_index("key")
file_locations
# %%
info_files_initial = {}
info_files_initial.update(
    {pth.name: pth
     for pth in base_info_initial_dir.rglob(f'./**/*.json') if pth.stem in list_of_selected_images})
pprint(dict(list(info_files_initial.items())[:5]))
print(f'Found {len(info_files_initial)} info files!')
# %%
info_files_additional = {}
list_of_selected_images_stems = [file.split(".")[0] for file in list_of_selected_images]
info_files_additional.update(
    {pth.name: pth
     for pth in base_info_custom_dir.rglob(f'./**/*.xml') if pth.stem in list_of_selected_images_stems})
pprint(dict(list(info_files_additional.items())[:5]))
print(f'Found {len(info_files_additional)} info files!')

# %%
all_bboxes = []
for key, file in info_files_initial.items():
    key_name = '.'.join(key.split(".")[:-1])
    info = json.load(io.open(file))
    xmin, ymin, xmax, ymax = info.get("Annotations")[0].get("BoundingBox")
    all_bboxes.append({
        "key": key_name,
        "xmin": int(xmin),
        "ymin": int(ymin),
        "xmax": int(xmax),
        "ymax": int(ymax),
        "info": file,
    })

for key, file in info_files_additional.items():
    root = ET.parse(file).getroot()
    key_name = root.findtext('./filename')
    info = root.findall("./object")[0]
    bbox = info.find("./bndbox")
    xmin = int(bbox.findtext("xmin"))
    ymin = int(bbox.findtext("ymin"))
    xmax = int(bbox.findtext("xmax"))
    ymax = int(bbox.findtext("ymax"))
    all_bboxes.append({
        "key": key_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "info": file,
    })
print(f"First five")
pprint(all_bboxes[:5])
print(f"Last five")
pprint(all_bboxes[-5:])
# %%
df_bbox = pd.DataFrame(all_bboxes).set_index('key', drop=False)
df_bbox
# %%
full_dataset = dataset.join(df_bbox).join(file_locations)
full_dataset
# %%
train_dataset, test_dataset = train_test_split(full_dataset, test_size=.2)
list_of_subsets = []
all_groups = train_dataset.groupby(["Mask_on", "Race", "Sex"])
min_val = all_groups.count().Person_num.min()
print(f"Number of class minimally available: {min_val}")
for category, grp in all_groups:
    list_of_subsets.append(grp.iloc[:min_val])
debiased_dataset = pd.concat(list_of_subsets)
biased_dataset = train_dataset[train_dataset.index.isin(dataset.index.difference(debiased_dataset.index))]
# %%
test_dataset
# %%
biased_dataset
# %%
debiased_dataset

# %%
data_root_dir = pathlib.Path('datasets')
data_raw_info_dir = data_root_dir / 'raw_info'
biased_dir = data_root_dir / 'data' / 'biased'
balanced_dir = data_root_dir / 'data' / 'debiased'
test_dir = data_root_dir / 'data' / 'test'
biased_dataset.to_csv('csv_datasets/biased.csv')
debiased_dataset.to_csv('csv_datasets/debiased.csv')
test_dataset.to_csv('csv_datasets/test.csv')


# %%
def generate_dataset(dataset: pd.DataFrame, target_dir: pathlib.Path, raw_info_dir):
    cols = ["file", "info", "xmin", "ymin", "xmax", "ymax", "Mask_on"]
    if not target_dir.exists():
        os.makedirs(target_dir.absolute())
    else:
        shutil.rmtree(target_dir.absolute())
        os.makedirs(target_dir.absolute())

    # Create train.txt
    rel_base_path = target_dir.parent.relative_to(target_dir.parent.parent) / target_dir.name

    # Copy files and generate the annotations
    with io.open(target_dir.parent / f'train_{target_dir.name}.txt', 'w') as train_text_file:
        for idx, row in tqdm(dataset.loc[:, cols].iterrows(), total=len(dataset)):
            file, info, xmin, ymin, xmax, ymax, label = row
            source_file_path = pathlib.Path(file).absolute()
            img = Image.open(source_file_path)
            w, h = img.size
            source_info_path = pathlib.Path(info).absolute()
            filename = source_file_path.name
            infoname = source_info_path.name
            annoname = source_file_path.stem + '.txt'
            file_destination = (target_dir / filename)
            info_destination = (raw_info_dir / infoname)
            anno_destination = (file_destination.parent / annoname)

            # label xcenter ycenter width height
            width = np.abs(xmax - xmin) / w
            height = np.abs(ymax - ymin) / h
            xcenter = ((xmax + xmin) / 2) / w
            ycenter = ((ymax + ymin) / 2) / h
            if not file_destination.parent.exists():
                print("==========================")
                print(f"{source_file_path}")
                print(f"{file_destination}")
                print(f"{anno_destination}")
                os.makedirs(file_destination.parent)

            if not info_destination.parent.exists():
                print(f"{source_info_path} -> {info_destination}")
                os.makedirs(info_destination.parent)

            dataset.loc[idx, "file"] = shutil.copy(source_file_path, file_destination)
            dataset.loc[idx, "info"] = shutil.copy(source_info_path, info_destination)
            dataset.loc[idx, "xcenter"] = xcenter
            dataset.loc[idx, "ycenter"] = ycenter
            dataset.loc[idx, "width"] = width
            dataset.loc[idx, "height"] = height
            train_text_file.write(f"{(rel_base_path / filename).as_posix()}\n")

            annotation_file = io.open(anno_destination, 'w')
            anno_destination.write_text(f"{label} {xcenter} {ycenter} {width} {height}")
            annotation_file.close()

    dataset.to_csv(target_dir.parent / (rel_base_path.name + '_dataset.csv'))
    #

    # Create obj.data
    with io.open(target_dir.parent.parent / f'object_{target_dir.name}.data', 'w') as object_data_file:
        content = {
            "classes": dataset["Mask_on"].nunique(),
            "train": (rel_base_path.parent / f'train_{target_dir.name}.txt').as_posix(),
            "valid": (rel_base_path.parent / 'train_test.txt').as_posix(),
            "names": (rel_base_path.parent / "obj.names").as_posix(),
            "backup": f"backup_{target_dir.name}/",
        }
        for key, val in content.items():
            object_data_file.write(f"{key} = {val}\n")

    # Update and save dataset


generate_dataset(test_dataset.copy(), test_dir, data_raw_info_dir)
generate_dataset(debiased_dataset.copy(), balanced_dir, data_raw_info_dir)
generate_dataset(biased_dataset.copy(), biased_dir, data_raw_info_dir)
# %%
