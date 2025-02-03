import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
import torchvision.transforms as TF
from torchvision.transforms import functional as F
import random
from PIL import Image

num_classes = 7
ST_COLORMAP = [
    [255, 255, 255],  
    [0, 0, 255],
    [128, 128, 128],
    [0, 128, 0],
    [0, 255, 0],
    [128, 0, 0],
    [255, 0, 0],
]
ST_CLASSES = [
    "unchanged",
    "water",
    "ground",
    "low vegetation",
    "tree",
    "building",
    "playground",  
]

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A = np.array([48.30, 46.27, 48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B = np.array([49.41, 47.01, 47.94])

dataset_root = ""

colormap2label = np.zeros(256**3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype="uint8")
    x = np.asarray(pred, dtype="int32")
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, time="A"):
    assert time in ["A", "B"]
    if time == "A":
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im


def normalize_images(imgs, time="A"):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def read_RSimages(mode):
    if mode == "train":
        with open("datasets/train_list_SECOND.txt", "r") as f:
            filename_list = f.read().split()
    else:
        with open("datasets/val_list_SECOND.txt", "r") as f:
            filename_list = f.read().split()

    imgs_list_A, imgs_list_B, labels_list_A, labels_list_B = [], [], [], []
    for i, filename in enumerate(filename_list):
        imgs_list_A.append(dataset_root + "/im1/" + filename)
        imgs_list_B.append(dataset_root + "/im2/" + filename)
        labels_list_A.append(dataset_root + "/label1/" + filename)
        labels_list_B.append(dataset_root + "/label2/" + filename)

    print(str(len(imgs_list_A)) + " " + mode + " images" + " loaded.")
    return imgs_list_A, imgs_list_B, labels_list_A, labels_list_B


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_list_A, self.labels_list_B = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, "A")
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, "B")

        # color → index
        label_A = io.imread(self.labels_list_A[idx])
        label_B = io.imread(self.labels_list_B[idx])
        label_A = Color2Index(label_A)
        label_B = Color2Index(label_B)  

        if self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)

        # PIL to Tensor
        img_A = F.to_tensor(img_A)
        img_B = F.to_tensor(img_B)
        label_A = torch.from_numpy(label_A)
        label_B = torch.from_numpy(label_B)

        return img_A, img_B, label_A, label_B

    def __len__(self):
        return len(self.imgs_list_A)
