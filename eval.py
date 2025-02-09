from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import SECOND, Landsat_SCD
from utils.utils import OA_mIoU_OAseg_Fscd_SeK
from models.mtcan import MTCAN

def main(args):
    if args.dataset == "SECOND":
        num_classes = SECOND.num_classes
        dataset = SECOND.Data(mode="val", random_flip=False)
        use_valid_mask = False
        model = MTCAN(num_classes, fam_mode="s234d8")
    elif args.dataset == "LandsatSCD":
        num_classes = Landsat_SCD.num_classes
        dataset = Landsat_SCD.Data(mode="val", random_flip=False)
        use_valid_mask = True
        model = MTCAN(num_classes, fam_mode="s1234d4")
    else:
        raise NotImplementedError()
    val_loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"), strict=False)
    model.cuda()

    with torch.no_grad():
        eval(val_loader, model, num_classes, use_valid_mask)


def eval(val_loader, model, num_classes, use_valid_mask=False):
    model.eval()
    torch.cuda.empty_cache()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        if use_valid_mask:
            imgs_A, imgs_B, labels_A, labels_B, valid_mask = data
        else:
            imgs_A, imgs_B, labels_A, labels_B = data

        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()

        out_change, outputs_A, outputs_B = model(imgs_A, imgs_B)
        out_change = torch.sigmoid(out_change)
        outputs_A = F.softmax(outputs_A, dim=1)
        outputs_B = F.softmax(outputs_B, dim=1)

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()

        change_mask = out_change.cpu().detach() > 0.5  # (B,1,H,W)
        change_mask = change_mask.squeeze(1)  # (B,H,W)
        if use_valid_mask:
            change_mask *= valid_mask
        preds_A = torch.argmax(outputs_A, dim=1).cpu().detach()  # (B,H,W)
        preds_B = torch.argmax(outputs_B, dim=1).cpu().detach()

        # masking
        preds_A = (preds_A * change_mask.long()).numpy()
        preds_B = (preds_B * change_mask.long()).numpy()

        for pred_A, pred_B, label_A, label_B in zip(
            preds_A, preds_B, labels_A, labels_B
        ):
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)

    OA, mIoU, OAseg, Fscd, SeK = OA_mIoU_OAseg_Fscd_SeK(
        preds_all, labels_all, num_classes
    )

    print(
        f"OA: {OA*100:.2f}\nmIoU: {mIoU*100:.2f}\nOAseg: {OAseg*100:.2f}\nFscd: {Fscd* 100:.2f}\nSeK: {SeK* 100:.2f}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    # SECOND or LandsatSCD datasets and checkpoints
    parser.add_argument("--dataset", default="SECOND", type=str)
    parser.add_argument(
        "--ckpt_path",
        default="your-ckpt-save-path",
    )  

    args = parser.parse_args()

    main(args)
