import math
import random
import numpy as np
import cv2
import skimage
from skimage import transform as sktransf
from numpy import pad
import matplotlib.pyplot as plt

def rand_rot90_MCD(img1, img2, label1, label2):
    r = random.random()
    if r < 0.5:
        return img1, img2, label1, label2
    else:
        return np.rot90(img1).copy(), np.rot90(img2).copy(), np.rot90(label1).copy(), np.rot90(label2).copy()

def rand_flip_MCD(img1, img2, label1, label2):
    r = random.random()
    if r < 0.25:
        return img1, img2, label1, label2
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(label1, axis=0).copy(), np.flip(label2, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(label1, axis=1).copy(), np.flip(label2, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label1[::-1, ::-1].copy(), label2[::-1, ::-1].copy()

def rand_rot90_flip_MCD(img1, img2, label1, label2):
    img1, img2, label1, label2 = rand_rot90_MCD(img1, img2, label1, label2)
    return rand_flip_MCD(img1, img2, label1, label2)
