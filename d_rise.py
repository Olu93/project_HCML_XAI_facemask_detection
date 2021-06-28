# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
# import torch
from PIL import Image
from numpy import asarray
from scipy import spatial
from matplotlib.patches import Rectangle
from keras.preprocessing import image
import json
import pathlib
import io
import pandas as pd

# ### FUNCTION DEFs
# class Model_yolo():
#     def __init__(self):
#         self.model = torch.hub.load('/hpc/shared/uu_ics_music/cmoll_magenta/hcml/yolov3', 'custom',
#                                     source='local')  # or 'yolov3_spp', 'yolov3_tiny'
#         self.input_size = (224, 224)

#     def run_on_batch(self, x):
#         return self.model(x)


# %%
def generate_masks(N, s, p1, resolution=(224, 224)):
    cell_size = np.ceil(np.array(resolution) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *resolution))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + resolution[0], y:y + resolution[1]]
    masks = masks.reshape(-1, *resolution, 1)
    return masks


# def load_img_yolo(path):
#     img = image.load_img(path, target_size=model_yolo.input_size)
#     x = asarray(img)
#     # x = np.expand_dims(x, axis=0)
#     return img, x
def load_img_yolo(file, resolution=(224, 224)):
    img = image.load_img(file, target_size=resolution)
    x = asarray(img)
    # x = np.expand_dims(x, axis=0)
    return img, x


def cos_sim(Pt, Pj):
    sim = 1 - spatial.distance.cosine(Pt, Pj)
    return sim


def transform_detection_vec(results):
    D = results.join(pd.get_dummies(results["xmin xmax ymin ymax confidence class_id".split()].class_id))
    D = D.drop("class_id", axis=1)
    return D


def IoU(Lt, Lj):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(Lt[0], Lj[0])
    yA = max(Lt[1], Lj[1])
    xB = min(Lt[2], Lj[2])
    yB = min(Lt[3], Lj[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    LtArea = (Lt[2] - Lt[0] + 1) * (Lt[3] - Lt[1] + 1)
    LjArea = (Lj[2] - Lj[0] + 1) * (Lj[3] - Lj[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(LtArea + LjArea - interArea)
    # return the intersection over union value
    return iou


batch_size = 100


def explain_yolo(dt_results: pd.DataFrame, inp: np.ndarray, masks: np.ndarray):
    # result.pred
    Dt = transform_detection_vec(dt_results["xmin xmax ymin ymax confidence class_id".split()], model)
    # print(Dt)
    Dp = []
    # # Make sure multiplication is being done for correct axes
    masked = inp * masks
    # # Limit since otherwise takes too long. Let run with entire N on HPC
    for i in tqdm(range(0, N), desc='Explaining'):
        # print(masked[i])
        dp_result = model.run_on_batch(masked[i])
        Dp.append(transform_detection_vec(dp_result, model))

    # preds = np.concatenate(preds)
    # sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    # sal = sal / N / p1
    return Dt, Dp


def pairwise_sim(Dt, Dp):
    W = []
    for i in range(0, len(Dt)):
        wt = []
        dt = Dt[i]
        for j in range(0, len(Dp)):
            wj = []
            for k in range(0, len(Dp[j])):
                dj = Dp[j][k]
                iou = IoU(dt[:4], dj[:4])
                csim = cos_sim(dt[5], dj[5])
                Oj = dj[4]
                s_dt_dj = iou * csim * Oj
                wj.append(s_dt_dj)
            wt.append(0) if len(wj) == 0 else wt.append(max(wj))
        W.append(wt)
    return W


def cal_saliency(W, masks):
    H = []
    for i in range(0, len(W)):
        Ht = 0
        for j in range(0, len(masks)):
            Ht = Ht + (W[i][j] * masks[j])
        H.append(Ht)
    return H


### Explanation excution

#Init model
# model_yolo = Model_yolo()
# %%

results_path = pathlib.Path('./datasets/result.json').absolute()
results_json = json.load(io.open(results_path))
img1 = results_json[0]
filename = 'datasets' / pathlib.Path(img1.get("filename"))
detected_quasi_flat_objects = [
    dict((*obj.items(), *obj.get("relative_coordinates").items())) for obj in img1.get("objects")
]
detected_objects = pd.DataFrame(detected_quasi_flat_objects).drop("relative_coordinates", axis=1)
detected_objects

# %%
detected_objects["xmin"] = detected_objects["center_x"] - (detected_objects["width"] * 0.5)
detected_objects["ymin"] = detected_objects["center_y"] - (detected_objects["height"] * 0.5)
detected_objects["xmax"] = detected_objects["center_x"] + (detected_objects["width"] * 0.5)
detected_objects["ymax"] = detected_objects["center_y"] + (detected_objects["height"] * 0.5)
detected_objects
# %%

# %%
img, x = load_img_yolo(filename)
img
# %%
# print(results.pandas().xyxy[0])

#Generate masks
N = 2000
s = 8
p1 = 0.5
masks = generate_masks(2000, 8, 0.5)

# %%
### Calc Dt and Dp for one image
Dt, Dp = explain_yolo(df, x, masks)

#Compute weights W for each mask for each dt from Dt and generate saliency maps Hs from W and masks
W = pairwise_sim(Dt, Dp)
Hs = cal_saliency(W, masks)

#Test
sal_maps = []
for i in range(0, len(Hs)):

    sal_map = Hs[i].reshape(model_yolo.input_size)
    bb = results.xyxy[0][i][:4].tolist()
    rect = Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=1, edgecolor='r', facecolor='none')
    plt.imshow(img)
    plt.imshow(sal_map, cmap='jet', alpha=0.5)
    ax = plt.gca()
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.savefig(str(i) + ".png")
    plt.cla()

# %%
