# %%
import os
from typing import Dict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pathlib
from IPython.display import display
import cv2
import tqdm.auto as tqdm
from pprint import pprint
from PIL import Image, ExifTags
import exifread
import json

# %%
root_path = pathlib.Path('.')
data_path = root_path / 'data'
dataset_biased = pd.read_csv(data_path / 'biased_dataset.csv').set_index('Img', drop=False)
dataset_debiased = pd.read_csv(data_path / 'debiased_dataset.csv').set_index('Img', drop=False)
dataset_test = pd.read_csv(data_path / 'test_dataset.csv').set_index('Img', drop=False)
dataset_augmented = pd.read_csv(data_path / 'debiased_augmented.csv').set_index('Img', drop=False)
# predictions_biased = json.load((data_path / 'biased_predictions.json').open())
# predictions_debiased = json.load((data_path / 'debiased_predictions.json').open())
# predictions_augmented = json.load((data_path / 'debiased_predictions.json').open())
predictions_test = json.load((data_path / 'test_predictions.json').open())
cols = set(
    "Mask_on, xcenter, ycenter, width, height, Race, Sex, SkinColor, Person_num, xmin, ymin, xmax, ymax".split(", "))
cols_protected = set("Race Sex SkinColor".split())
cols_label = set(["Mask_on"])
cols_id = set(["Img"])
cols_interest = cols_protected.union(cols_label).union(cols_id)
cols_others = cols - cols_interest
cols_interactions = set(["Race_Sex"])

label_mapping = {
    "with_mask": 1,
    "without_mask": 0,
    "not_detected": np.nan,
}

race_mapping = {
    "South-Asian": 4,
    "Asian": 3,
    "Black": 2,
    "Other": 1,
    "Caucasian": 0,
}

sex_mapping = {"Male": 1, "Female": 0}
skincolor_mapping = {"White": 0, "Brown": 1, "Black": 2}

rev_label_mapping = {val: key for key, val in label_mapping.items()}
rev_race_mapping = {val: key for key, val in race_mapping.items()}
rev_sex_mapping = {val: key for key, val in sex_mapping.items()}
rev_skincolor_mapping = {val: key for key, val in skincolor_mapping.items()}

all_mappings = {
    "Race": rev_race_mapping,
    "Mask_on": rev_label_mapping,
    "Sex": rev_sex_mapping,
    "SkinColor": rev_skincolor_mapping,
}


def replace_codes(dataset: pd.DataFrame, mapping: Dict[str, Dict[int, str]]):
    return dataset.replace(mapping)


def add_interaction_cols(dataset: pd.DataFrame, col1: str, col2: str):
    dataset[f"{col1}_{col2}"] = dataset[col1] + "-" + dataset[col2]
    return dataset


def extract_info(pth: pathlib.Path) -> dict:
    # # exclude = ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote')
    # exclude = ()
    # key = pth.name
    # # img = Image.open()
    # tags = exifread.process_file(pth.relative_to("datasets").open("rb"))
    # all_exifs_data = {key: val for key, val in tags.items() if key not in exclude}
    # result = {}
    # result["Img"] = key
    # result.update(all_exifs_data)
    # print(all_exifs_data)

    key = pth.name
    img = Image.open(pathlib.Path(*pth.parts[1:]))
    all_exifs_data = {key: val for key, val in img.info.items()}
    result = {}
    result["Img"] = key
    result.update(all_exifs_data)
    # print(all_exifs_data)

    return result


def get_resolutions(dataset: pd.DataFrame) -> pd.DataFrame:
    col_resolutions = list(
        map(lambda s: f"info_{s.strip()}",
            "im_key, im_width, im_height, im_channel, im_colors, im_resolution, im_aspect_ratio".split(",")))
    dataset_paths = dataset.file
    path_data = [pathlib.Path(pth) for pth in dataset_paths]
    info_data = [extract_info(pth) for pth in path_data]
    df_info = pd.DataFrame(info_data).set_index("Img")
    img_data_dict = {
        pth.name: cv2.imread(filename=str(pathlib.Path(*pth.parts[1:])))
        for pth in tqdm.tqdm(path_data, total=len(path_data))
    }
    luminance_dict = [(key, val.sum(axis=-1) / val.shape[-1]) for key, val in img_data_dict.items()]
    dim_data_dict = {key: img.shape for key, img in img_data_dict.items()}
    resolution_data_dict = {
        key: (*val, "RGB" if val[2] > 1 else "Grey", val[0] * val[1], val[0] / val[1])
        for key, val in dim_data_dict.items()
    }
    resolution_data = ((im_key, ) + im_vals for im_key, im_vals in resolution_data_dict.items())
    df_resolution = pd.DataFrame(resolution_data, columns=col_resolutions).set_index(col_resolutions[0])
    df_luminance = pd.DataFrame(luminance_dict, columns="Img info_im_luminance".split()).set_index("Img")
    # display(df_info)
    result = dataset.join(df_resolution).join(df_info).join(df_luminance)
    ser_res = result["info_im_resolution"]

    result["info_im_normed_resolution"] = (ser_res - ser_res.mean()) / ser_res.std()
    result["info_im_scaled_resolution"] = (ser_res - ser_res.min()) / (ser_res.max() - ser_res.min())
    # result["info_im_lumincence"] = result[(f"info_im_{val}" for val in "width height")]
    return result


def join_true_and_predicted(dataset: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    cols = [
        'Img',
        'pred_class_id',
        'pred_name',
        'pred_center_x',
        'pred_center_y',
        'pred_width',
        'pred_height',
        'pred_confidence',
        'pred_num_detections',
        'pred_detections',
    ]
    precols = ['frame_id' 'filename' 'objects']
    df = pd.DataFrame(predictions, columns=precols, dtype=np.int64)
    df.index = df["filename"].map(lambda idx: pathlib.Path(idx).name)
    i = 0

    for (f, img_detections) in tqdm.tqdm(zip(df.index, df.objects)):
        # try:

        df.loc[f] = [
            f,
            img_detections[0]['class_id'],
            img_detections[0]['name'],
            img_detections[0]['relative_coordinates']['center_x'],
            img_detections[0]['relative_coordinates']['center_y'],
            img_detections[0]['relative_coordinates']['width'],
            img_detections[0]['relative_coordinates']['height'],
            img_detections[0]['confidence'],
            len(img_detections),
            (int("".join([detection.get("class_id") for detection in img_detections])) if len(img_detections) else 0),
        ]
        # except:
        #     df.loc[f] = [os.path.basename(f), None, None, None, None, None, None, ]
        # i += 1
    df = dataset.merge(df, on=['Img'], how='left')

    return df


def prepare_dataset(dataset: pd.DataFrame, results_json: dict, all_mappings: Dict[str, Dict[int, str]]) -> pd.DataFrame:
    dataset = join_true_and_predicted(dataset, results_json)
    dataset = replace_codes(dataset, all_mappings)
    dataset = add_interaction_cols(dataset, "Race", "Sex")
    dataset = get_resolutions(dataset)

    return dataset


# dataset_debiased = prepare_dataset(dataset_debiased, all_mappings)
# dataset_biased = prepare_dataset(dataset_biased, all_mappings)
# dataset_augmented = prepare_dataset(dataset_augmented, all_mappings)
dataset_test = prepare_dataset(dataset_test, predictions_test, all_mappings)
dataset_test
# %%
print("Dataset: biased")
display(dataset_biased[cols].sample(5))

print("Dataset: debiased")
display(dataset_debiased[cols].sample(5))

print("Dataset: test")
display(dataset_test[cols].sample(5))

print("Dataset: augmented")
display(dataset_augmented[cols].sample(5))

# %%
# dataset_test.melt(cols_interest, cols_others, var_name="x",
#                   value_name="y").melt(cols_interest.union(set(("x", "y"))) - cols_label.union(cols_id), cols_protected)
# melted_test = dataset_test.melt(id_vars=cols_id.union(cols_label), value_vars="Img", value_name="protected")
# sns.catplot(data=melted_test, x="protected", hue="Mask_on", kind="count")
# plt.show()
# # %%
# tmp = dataset_test.melt(cols_interest, cols_others, var_name="x", value_name="y")
# # %%
# tmp = dataset_test.melt(cols.union(("Sex", )) - cols_protected.difference(("Sex", )),
#                         cols_protected - set(("Sex", )),
#                         var_name="protected",
#                         value_name="y")
# tmp
# groups = tmp.groupby("Mask_on")
# fig, axes = plt.subplots(1, len(groups), figsize=(12, 4))
# for (idx, grp), ax in zip(groups, axes):
#     sns.histplot(data=grp, x=("y", ), ax=ax)
# plt.show()

# %%
PROTECTED_ATTRIBUTE = "Protected Attribute"
tmp = dataset_test[cols].groupby(list(cols_interest - cols_id))["Person_num"].count().reset_index().melt(
    ("Mask_on", "Person_num"), cols_protected, var_name=PROTECTED_ATTRIBUTE)
tmp["Value"] = tmp[PROTECTED_ATTRIBUTE] + ":" + tmp["value"]
tmp


# %%
# tmp = dataset_test[cols].groupby(list(cols_interest - cols_id))["Person_num"].count().reset_index()
def decorate_graph(g: plt.Axes, ylabel: str):
    g.set_ylabel(ylabel)
    return g


fig, axes = plt.subplots(2, 3, figsize=(15, 7))
print(axes.shape)


def generate_graph(df, all_mappings, row, ax, col, lbl_col):
    mask = df[df[lbl_col] == all_mappings[lbl_col][row]]
    g = sns.histplot(
        data=mask,
        ax=ax,
        x=col,
    )
    return decorate_graph(g, all_mappings[lbl_col][row])


def get_data_subset(all_mappings, tmp, grp, lbl_col):
    return grp[grp[lbl_col] == all_mappings[lbl_col][tmp]]


for idx, col in enumerate(cols_protected):
    grps = dataset_test.groupby(col)
    for index, grp in grps:
        lbl_col = list(cols_label)[0]
        row = 0
        df = get_data_subset(all_mappings, row, grp, lbl_col)
        ax = axes[row, idx]
        g = generate_graph(grp, all_mappings, row, ax, col, lbl_col)
        row = 1
        df = get_data_subset(all_mappings, row, grp, lbl_col)
        ax = axes[row, idx]
        g = generate_graph(grp, all_mappings, row, ax, col, lbl_col)

plt.show()
# g.despine(left=True)
# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")
# plt.show()
# %%
