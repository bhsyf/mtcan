import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def weighted_BCE(logit_pixel, truth_pixel, weight_pos=0.25):
    logit = logit_pixel.view(-1)  # (BHW,)
    truth = truth_pixel.view(-1)
    weight_neg = 1-weight_pos
    assert logit.shape == truth.shape

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction="none")

    pos = (truth > 0.5).float() 
    neg = (truth <= 0.5).float() 
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()
    return loss


def weighted_BCE_mask(logit, gt, valid_mask, weight_pos=0.25):
    """logit,gt,valid_mask shape: (B,1,H,W),
    用valid_mask把未变化和空白分开"""
    weight_neg = 1 - weight_pos
    assert logit.shape == gt.shape
    pos_mask = (gt > 0.5).float()
    neg_mask = (1 - gt) * valid_mask
    pos_num = pos_mask.sum().item() + 1e-12
    neg_num = neg_mask.sum().item() + 1e-12
    loss = F.binary_cross_entropy_with_logits(logit, gt, reduction="none")
    loss = (
        weight_pos * pos_mask * loss / pos_num + weight_neg * neg_mask * loss / neg_num
    ).sum()
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss_f = nn.CosineEmbeddingLoss(reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)  # (B,nC,H,W)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)  # (B,H,W,nC)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])  # (N,nC)
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()  # unchange mask(bool)
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss


class ContrastiveLossWithMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_f = nn.CosineEmbeddingLoss(reduction="none")

    def forward(self, x1, x2, label_change, mask):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)  # (B,nC,H,W)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)  # (B,H,W,nC)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])  # (N,nC)
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()  # unchange mask(bool)
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])
        mask = mask.reshape(-1)

        loss = self.loss_f(x1, x2, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss
