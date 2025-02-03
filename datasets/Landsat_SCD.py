import os
from typing import Any
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
from tqdm import tqdm

dataset_root = ""
num_classes = 5
CLASSES = ["0: No change", "1: Farmland", "2: Desert", "3: Building", "4: Water"]
COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]

# my, mask blank areas
MEAN_A0 = np.array([92.31, 88.99, 86.87])
STD_A0 = np.array([35.85, 35.29, 34.17])
MEAN_B0 = np.array([86.32, 85.13, 83.18])
STD_B0 = np.array([40.00, 38.86, 38.53])

# my, without masking blank areas
MEAN_A1 = np.array([141.66, 139.35, 137.88])
STD_A1 = np.array([64.56, 64.98, 65.04])
MEAN_B1 = np.array([137.50, 136.66, 135.30])
STD_B1 = np.array([68.65, 68.16, 68.49])


colormap2label = np.zeros(256**3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = np.asarray(ColorLabel, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap.astype(np.int8)


def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype="uint8")
    x = np.asarray(pred, dtype="int32")
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, mean, std):
    return (im - mean) / std


def normalize_images(imgs, time="A"):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def norm_only_nonblank_areas(imgA, imgB):
    """blank areas → RGB[0,0,0],
        nonblank areas → norm"""
    invalid_mask = np.all(imgA == 255, axis=2)
    valid_mask = np.logical_not(invalid_mask)
    valid_mask = np.expand_dims(valid_mask, axis=-1)
    norm_imgA = (imgA - MEAN_A0) / STD_A0
    norm_imgB = (imgB - MEAN_B0) / STD_B0
    norm_imgA *= valid_mask
    norm_imgB *= valid_mask
    return norm_imgA, norm_imgB


def read_RSimages(mode):
    assert mode in ["train", "val"]
    if mode == "train":
        list_path = "datasets/train_list_Landsat-SCD.txt"
    else:
        list_path = "datasets/val_list_Landsat-SCD.txt"
    img_A_dir = os.path.join(dataset_root, "A")
    img_B_dir = os.path.join(dataset_root, "B")
    # index label
    label_A_dir = os.path.join(dataset_root, "labelA_index")
    label_B_dir = os.path.join(dataset_root, "labelB_index")

    list_info = open(list_path, "r")
    data_list = list_info.readlines()
    data_list = [item.rstrip() for item in data_list]

    imgsA_list, imgsB_list, labelsA_list, labelsB_list = [], [], [], []
    nums = len(data_list)
    for it in data_list:
        if it[-4:] == ".png":
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            imgsA_list.append(img_A_path)
            imgsB_list.append(img_B_path)
            labelsA_list.append(label_A_path)
            labelsB_list.append(label_B_path)
    print(f"{nums} {mode} images loaded.")
    return imgsA_list, imgsB_list, labelsA_list, labelsB_list


class Data(data.Dataset):
    def __init__(
        self, mode, random_flip=False, only_norm_nonblank=True, use_valid_mask=True
    ):
        super().__init__()
        self.num_classes = 5
        self.random_flip_rot = random_flip
        self.use_valid_mask = use_valid_mask
        self.only_norm_nonblank = only_norm_nonblank
        self.imgs_list_A, self.imgs_list_B, self.labels_list_A, self.labels_list_B = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgsA_list[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgsA_list[idx])
        img_B = io.imread(self.imgsB_list[idx])
        label_A = io.imread(self.labelsA_list[idx])
        label_B = io.imread(self.labelsB_list[idx])

        if self.random_flip_rot:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(
                img_A, img_B, label_A, label_B)

        if self.use_valid_mask:
            invalid_mask = np.all(img_A == 255, axis=2)
            valid_mask = np.logical_not(invalid_mask)
            valid_mask = torch.from_numpy(valid_mask)

        # norm
        if self.only_norm_nonblank:
            img_A, img_B = norm_only_nonblank_areas(img_A, img_B)
        else:
            img_A = normalize_image(img_A, MEAN_A1, STD_A1)
            img_B = normalize_image(img_B, MEAN_B1, STD_B1)

        # PIL to Tensor
        img_A = F.to_tensor(img_A)
        img_B = F.to_tensor(img_B)
        label_A = torch.from_numpy(label_A)
        label_B = torch.from_numpy(label_B)

        if self.use_valid_mask:
            return img_A, img_B, label_A, label_B, valid_mask
        else:
            return img_A, img_B, label_A, label_B

    def __len__(self):
        return len(self.imgsA_list)



