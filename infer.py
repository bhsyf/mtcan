import os
import time
import numpy as np
import torch
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets import Landsat_SCD
from datasets import SECOND
from models.mtcan import MTCAN

def main(args):
    begin_time = time.time()
    if args.dataset == "second":
        dataset = SECOND.Data(mode="test")
        num_classes = SECOND.num_classes
    elif args.dataset == "landsat":
        dataset = Landsat_SCD.Data(mode="test")
        num_classes = Landsat_SCD.num_classes
    else:
        raise NotImplementedError("dataset not supported")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batchsize must be 1

    net = MTCAN(num_classes)
    net.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device("cpu")))
    net.cuda().eval()

    with torch.no_grad():
        predict(args, net, dataset, dataloader)

    time_use = time.time() - begin_time
    print("Total time: %.2fs" % time_use)


def predict(args, net, dataset, dataloader):
    pred_A_dir_rgb = os.path.join(args.save_path, "im1_rgb")
    pred_B_dir_rgb = os.path.join(args.save_path, "im2_rgb")
    if not os.path.exists(pred_A_dir_rgb):
        os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb):
        os.makedirs(pred_B_dir_rgb)

    for vi, data in enumerate(dataloader):
        if args.dataset == "second":
            imgs_A, imgs_B, _, _ = data
        elif args.dataset == "landsat":
            imgs_A, imgs_B, _, _, valid_mask = data

        imgs_A, imgs_B, valid_mask = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = dataset.get_mask_name(vi)  

        out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

        out_change = torch.sigmoid(out_change)  # prob
        outputs_A = F.softmax(outputs_A, dim=1)
        outputs_B = F.softmax(outputs_B, dim=1)

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze(1) * valid_mask  # (B,H,W)
        change_mask = change_mask.squeeze(0)  # (H,W)
        pred_A = torch.argmax(outputs_A, dim=1).squeeze(0)
        pred_B = torch.argmax(outputs_B, dim=1).squeeze(0)

        pred_A = (pred_A * change_mask.long()).numpy()
        pred_B = (pred_B * change_mask.long()).numpy()

        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)

        if args.dataset == "second":
            io.imsave(pred_A_path, SECOND.Index2Color(pred_A))
            io.imsave(pred_B_path, SECOND.Index2Color(pred_B))
        elif args.dataset == "landsat":
            io.imsave(pred_A_path, Landsat_SCD.Index2Color(pred_A))
            io.imsave(pred_B_path, Landsat_SCD.Index2Color(pred_B))
        print(pred_A_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="second", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--net", default="mtcan", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument("--save_path", default="/results-save-path", type=str)
    
    args = parser.parse_args()
    main(args)
