import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from skimage.transform import resize
from tqdm import tqdm
from PIL import Image
from numpy import asarray, matrix
from scipy import spatial
import scipy.spatial as sp
from matplotlib.patches import Rectangle
# from keras.preprocessing import image
import sys
import pathlib
import cv2
import pandas as pd
from pprint import pprint

DARKNET_PATH = './../../misc/darknet'
DATASET_PATH = './'
# DARKNET_PATH = './darknet/'
# DATASET_PATH = './datasets/datasets/'
DARKNET = True
if DARKNET:
    pass
    # sys.path.insert(0, str(pathlib.Path(DARKNET_PATH).resolve()))
    # import darknet
    # from darknet import network_height, network_width, load_network, print_detections, detect_image, decode_detection, load_meta, load_image
    # from darknet_images import batch_detection, save_annotations, image_detection
    # config_file = str(pathlib.Path(DATASET_PATH + 'yolo-obj.cfg').absolute())
    # weights_file = str(pathlib.Path(DATASET_PATH + 'weights/biased.weights').absolute())
    # data_file = str(pathlib.Path(DATASET_PATH + 'object_biased.data').absolute())
    # meta = load_meta(data_file.encode('ascii'))
    # network, class_names, colors = load_network(config_file, data_file, weights_file, batch_size=1)
    # network
else:
    import torch

    ### FUNCTION DEFs
    class Model_yolo():
        def __init__(self):
            self.model = torch.hub.load('/hpc/shared/uu_ics_music/cmoll_magenta/hcml/yolov3', 'custom',
                                        source='local')  # or 'yolov3_spp', 'yolov3_tiny'
            self.input_size = (224, 224)

        def run_on_batch(self, x):
            return self.model(x)


# def generate_masks(N, s, p1):
#     cell_size = np.ceil(np.array(model_yolo.input_size) / s)
#     up_size = (s + 1) * cell_size

#     grid = np.random.rand(N, s, s) < p1
#     grid = grid.astype('float32')

#     masks = np.empty((N, *model_yolo.input_size))

#     for i in tqdm(range(N), desc='Generating masks'):
#         # Random shifts
#         x = np.random.randint(0, cell_size[0])
#         y = np.random.randint(0, cell_size[1])
#         # Linear upsampling and cropping
#         masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
#                                 anti_aliasing=False)[x:x + model_yolo.input_size[0], y:y + model_yolo.input_size[1]]
#     masks = masks.reshape(-1, *model_yolo.input_size, 1)
#     return masks


def generate_masks(N, s, p1, resolution):
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


def load_img_yolo(path):
    img = image.load_img(path, target_size=model_yolo.input_size)
    x = asarray(img)
    # x = np.expand_dims(x, axis=0)
    return img, x


def cos_sim(Pt, Pj):
    sim = 1 - spatial.distance.cosine(Pt, Pj)
    return sim


def transform_detection_vec(results, model):
    D = []
    for i in range(0, len(results.pred[0])):
        pos_i = int(results.pred[0][i][5])
        P_i = results.pred[0][i].tolist()
        P_onehot = np.zeros(model.model.nc)
        P_onehot[pos_i] = 1
        P_i[5] = P_onehot
        D.append(P_i)
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


# def explain_yolo(model, inp, masks):
#     dt_results = model.run_on_batch(inp)
#     # result.pred
#     Dt = transform_detection_vec(dt_results, model)
#     # print(Dt)
#     Dp = []
#     # # Make sure multiplication is being done for correct axes
#     masked = inp * masks
#     # # Limit since otherwise takes too long. Let run with entire N on HPC
#     for i in tqdm(range(0, N), desc='Explaining'):
#         # print(masked[i])
#         dp_result = model.run_on_batch(masked[i])
#         Dp.append(transform_detection_vec(dp_result, model))


#     # preds = np.concatenate(preds)
#     # sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
#     # sal = sal / N / p1
#     return Dt, Dp
def resize_img(img, width, height):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img_resized


def generate_Dp(network, class_names, colors, cv2_img):
    # dt_results = model.run_on_batch(inp)
    # result.pred
    # Dt = transform_detection_vec(dt_results, model)
    # print(Dt)
    Dp = []
    # # Make sure multiplication is being done for correct axes
    # masked = inp * masks
    # # Limit since otherwise takes too long. Let run with entire N on HPC
    N = 2000
    s = 8
    p1 = 0.5
    # masks = generate_masks(2000, 8, 0.5)
    width, height, channel = cv2_img.shape
    # width = network_width(network)
    # height = network_height(network)
    # img_resized = cv2.resize(cv2_img, (width, height))
    # masks = generate_masks(N, s, p1, (img_shape[0], img_shape[1]))
    masks = generate_masks(N, s, p1, (width, height))
    masked = cv2_img * masks
    masked_adjusted = np.float32(masked).astype(np.uint8)
    # print(type(masked_float32[1][0][0][0]))
    # print(masked_float32.astype(np.uint8))
    # imgs, batch_preds = batch_detection(network, masked_adjusted[:16], class_names, colors, batch_size=8)
    # print(len(batch_preds), batch_preds[0])
    # print(len(imgs), len(batch_preds))
    for i in tqdm(range(0, N), desc='Explaining'):
        #     # print(masked[i])
        image, detections = image_detection_cv2(masked_adjusted[i], network, class_names, colors, 0.3)
        Dp.append(generate_D(image, detections, class_names))
    # print("OHBOOOOY ", Dp)
    return Dp, masks


# def pairwise_sim(Dt, Dp):
#     W = []

#     for i in range(0, len(Dt)):
#         wt = []
#         dt = Dt[i]
#         for j in range(0, len(Dp)):
#             wj = []
#             for k in range(0, len(Dp[j])):
#                 dj = Dp[j][k]
#                 iou = IoU(dt[:4], dj[:4])
#                 csim = cos_sim(dt[5], dj[5])
#                 Oj = dj[4]
#                 s_dt_dj = iou * csim * Oj
#                 wj.append(s_dt_dj)
#             wt.append(0) if len(wj) == 0 else wt.append(max(wj))
#         W.append(wt)
#     return W


def IoU_faf(Dt_bbox, Dp_bbox):

    # print(Dt_bbox.iloc[:, 0].shape, Dp_bbox.iloc[:, 0].shape)
    # print(Dt_bbox.shape, Dp_bbox.shape)
    xA = np.maximum(Dt_bbox.iloc[:, 0].values.reshape(1, -1), Dp_bbox.iloc[:, 0].values.reshape(-1, 1)).flatten()
    xB = np.maximum(Dt_bbox.iloc[:, 1].values.reshape(1, -1), Dp_bbox.iloc[:, 1].values.reshape(-1, 1)).flatten()
    yA = np.maximum(Dt_bbox.iloc[:, 2].values.reshape(1, -1), Dp_bbox.iloc[:, 2].values.reshape(-1, 1)).flatten()
    yB = np.maximum(Dt_bbox.iloc[:, 3].values.reshape(1, -1), Dp_bbox.iloc[:, 3].values.reshape(-1, 1)).flatten()
    # print(xA.shape)

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    LtArea = (Dt_bbox.iloc[:, 2] - Dt_bbox.iloc[:, 0] + 1) * (Dt_bbox.iloc[:, 3] - Dt_bbox.iloc[:, 1] + 1)
    LjArea = (Dp_bbox.iloc[:, 2] - Dp_bbox.iloc[:, 0] + 1) * (Dp_bbox.iloc[:, 3] - Dp_bbox.iloc[:, 1] + 1)
    LAdds = np.add(LtArea.values.reshape(1, -1), LjArea.values.reshape(-1, 1)).flatten()

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    # print(interArea.shape, LtArea.shape, LjArea.shape, LAdds.shape)

    iou = interArea / (LAdds - interArea)
    # print(iou)
    return iou


def conf_faf(Dt_conf, Dp_conf):
    repeats = len(Dt_conf)
    return np.tile(Dp_conf.values.flatten(), repeats)


def cosine_similarity_faf(Dt_onehot, Dp_onehot):
    matrix1 = Dt_onehot.values
    matrix2 = Dp_onehot.values
    return 1 - sp.distance.cdist(matrix1, matrix2, 'cosine').flatten()


def pairwise_sim(Dt, Dps):
    bbox_cols = "xmin xmax ymin ymax".split()
    confidence_cols = ["conf"]
    onehot_cols = [0, 1]
    all_weights = []
    len_Dt = len(Dt)
    # print(len_Dt)
    for Dp in Dps:
        bbox = IoU_faf(Dt[bbox_cols], Dp[bbox_cols])
        confs = conf_faf(Dt[confidence_cols], Dp[confidence_cols])
        cosines = cosine_similarity_faf(Dt[onehot_cols], Dp[onehot_cols])
        # print(Dt.shape, Dp.shape)
        # print(bbox.shape, confs.shape, cosines.shape)
        weights = bbox * confs * cosines
        reshaped_weights = weights.reshape(-1, len_Dt).T
        # print(reshaped_weights)
        all_weights.append(reshaped_weights)
        # print(bbox * confs * cosines)
    W = all_weights
    return W


def calc_saliency_faf(W, masks):
    len_Dt = W[0].shape[0]
    num_masks, width, height, channel = masks.shape
    some_sums = np.zeros((len_Dt, width, height))
    for dt_idx in range(len_Dt):
        for m_idx, mask in enumerate(masks):
            # print(mask.shape)
            # print(len(W))
            # print(len(W[0]))
            # print(W[m_idx].shape)
            # print(W[m_idx][dt_idx].shape)
            # print(some_sums[dt_idx].shape)
            some_sums[dt_idx] = some_sums[dt_idx] + np.sum(W[m_idx][dt_idx] * mask, axis=2)

    return some_sums


def cal_saliency(W, masks):
    H = []
    for i in range(0, len(W)):
        Ht = 0
        for j in range(0, len(masks)):
            Ht = Ht + (W[i][j] * masks[j])
        H.append(Ht)
    return H


def generate_D(image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    # file_name = os.path.splitext(name)[0] + ".txt"
    # print("FILE NAME:", file_name)
    # with open(file_name, "w") as f:
    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf']
    D_list = []
    one_hot = np.zeros((len(detections), len(class_names)))
    for idx, (label, confidence, bbox) in enumerate(detections):
        # print("CLASS_NAMES LEN:",len(class_names))
        one_hot[idx, class_names.index(label)] = 1
        # print(one_hot)
        ### Prep bbox (xmin, ymin, xmax, ymax)
        x, y, w, h = bbox
        xmin, xmax = x - w / 2, y - h / 2
        ymin, ymax = x + w / 2, y + h / 2
        D_list.append([
            xmin,
            ymin,
            xmax,
            ymax,
            (float(confidence) / 100),
        ])

    D = pd.DataFrame(D_list, columns=columns).join(pd.DataFrame(one_hot))
    # print(Dt)
    # Dt = Dt.join(pd.get_dummies(Dt["onehot"])).drop("onehot", axis=1)
    return D


def image_detection_cv2(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    # image = cv2.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # image_cv2 = cv2.imread(image_path)
    # darknet_image = darknet.load_image(image_path.encode('ascii'), image_cv2.shape[1], image_cv2.shape[0])

    # image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    # image = darknet.draw_boxes(detections, image_cv2, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


import pickle
### Explanation excution
def generate_drise_masks(network, class_names, colors, images, thresh=.5, hier_thresh=.5, nms=.45):
    all_cv2_imgs = [cv2.imread(img) for img in images]
    resized_img = [resize_img(img, 300, 300) for img in all_cv2_imgs]
    all_detections = [image_detection_cv2(img, network, class_names, colors, 0.3) for img in resized_img]
    Dts = [(image, generate_D(image, detections, class_names)) for image, detections in all_detections]
    NDps = [generate_Dp(network, class_names, colors, image) for image, Dt in Dts]

    pickle.dump((Dts, NDps), open('./tmp.pkl', 'wb'))
    Ws = [(pairwise_sim(Dt, Dps), masks, image, Dt) for (image, Dt), (Dps, masks) in zip(Dts, NDps)]
    saliencies_per_image_per_detection = [(calc_saliency_faf(W, masks), image, Dt) for W, masks, image, Dt in Ws]

    # Weights

    # meta_data = [(*cv2.imread(img).shape, img) for img in images]
    # img_objs = [load_image(name.encode('ascii'), w, h) for h, w, c, name in meta_data]
    # detections = [detect_image(network, class_names, img, thresh, hier_thresh, nms) for img in img_objs]
    # decoded_detections = [decode_detection(detection) for detection in detections]
    return saliencies_per_image_per_detection

def generate_drise_masks_short(Dts, NDps):
    Ws = [(pairwise_sim(Dt, Dps), masks, image, Dt) for (image, Dt), (Dps, masks) in zip(Dts, NDps)]
    saliencies_per_image_per_detection = [(calc_saliency_faf(W, masks), image, Dt) for W, masks, image, Dt in Ws]
    return saliencies_per_image_per_detection


# saliency_maps_per_detection_img1, original_image, Dt = generate_drise_masks(network, class_names, colors,
#                                                                             ['./quicktest/2.jpg'])[0]
saliency_maps_per_detection_img1, original_image, Dt = generate_drise_masks_short(*pickle.load(open('./tmp.pkl', 'rb')))[0]
for idx, dt_saliency in enumerate(saliency_maps_per_detection_img1):
    # for i in range(0, len(Hs)):

    #     sal_map = Hs[i].reshape(model_yolo.input_size)
    bb = list(Dt.iloc[idx, :4])
    rect = Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=1, edgecolor='r', facecolor='none')
    plt.imshow(original_image)
    plt.imshow(dt_saliency, cmap='jet', alpha=0.5)
    ax = plt.gca()
    # Add the patch to the Axes
    ax.add_patch(rect)
    # plt.savefig(str(i) + ".png")
    plt.show()
    plt.cla()



















# images = load_images('datasets/datasets/quick_test_images/special.png')

# image, detections = image_detection(
#             'datasets/datasets/quick_test_images/special.png', network, class_names, colors, 0.3
#             )
# generate_D(image, detections, class_names)

# print(detections)
# img = cv2.imread('datasets/datasets/quick_test_images/body1.png')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# save_annotations("image_name", image, detections, class_names)

# print(load_meta(str(data_file).encode('UTF-8')))
# print(classify(network, meta, load_image(b'datasets/datasets/quick_test_images/0.jpg',224,224)))

# print()
# print(class_names)
#

# #Init model
# model_yolo = Model_yolo()

# img, x = load_img_yolo('body1.png')
# results = model_yolo.run_on_batch(x)
# print(results.pandas().xyxy[0])

# #Generate masks
# N = 2000
# s = 8
# p1 = 0.5
# masks = generate_masks(2000, 8, 0.5)

# ### Calc Dt and Dp for one image
# Dt, Dp = explain_yolo(model_yolo, x, masks)

# #Compute weights W for each mask for each dt from Dt and generate saliency maps Hs from W and masks
# W = pairwise_sim(Dt, Dp)
# Hs = cal_saliency(W, masks)

# #Test
# sal_maps = []
# for i in range(0, len(Hs)):

#     sal_map = Hs[i].reshape(model_yolo.input_size)
#     bb = results.xyxy[0][i][:4].tolist()
#     rect = Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=1, edgecolor='r', facecolor='none')
#     plt.imshow(img)
#     plt.imshow(sal_map, cmap='jet', alpha=0.5)
#     ax = plt.gca()
#     # Add the patch to the Axes
#     ax.add_patch(rect)
#     plt.savefig(str(i) + ".png")
#     plt.cla()
