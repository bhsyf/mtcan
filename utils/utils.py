import os
import math
import random
import numpy as np
from scipy import stats
import torch

def OA_mIoU_OAseg_Fscd_SeK(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(
            set([0, 1, 2, 3, 4, 5, 6])
        ), "unrecognized label number"
        label_array = np.array(label)
        assert (
            infer_array.shape == label_array.shape
        ), "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)

    # calc mIoU
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()

    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum() 
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum / pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP / change_pred_sum
    SC_Recall = SC_TP / change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])

    OA = np.diag(hist).sum() / pixel_sum
    OAseg = SC_TP / hist_fg.sum()
    return OA, IoU_mean, OAseg, Fscd, Sek
