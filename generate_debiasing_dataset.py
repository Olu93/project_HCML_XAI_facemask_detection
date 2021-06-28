# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
import shutil
import pathlib
# %%
all_dir = pathlib.Path('./all')
all_imgs_dir = all_dir / "imgs"
all_rinfo_dir = all_dir / "raw_info"
debiased_dataset_labels = pd.read_csv('./additional_dataset.csv', index_col=0)
debiased_dataset_labels

# %%
df = debiased_dataset_labels.loc[debiased_dataset_labels.Is_Complicated == False]
df = df.loc[df.Race != 1]
df
# %%
df_shuffled = df.sample(frac=1)
df_shuffled
# %%
list_of_subsets = []
all_groups = df.groupby(["Mask_on", "Race", "Sex"])
min_val = all_groups.count().Person_num.min()
print(f"Number of class minimally available: {min_val}")
# %%
for category, grp in all_groups:
    list_of_subsets.append(grp.iloc[:min_val])

# %%
debiased_dataset = pd.concat(list_of_subsets)
# %%
