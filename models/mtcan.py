import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from mmcv.cnn import NonLocal2d
from models.backbone import MSCAN
from collections import OrderedDict


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEM(NonLocal2d):
    def __init__(self, *arg, temperature=0.05, **kwargs):
        super().__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= torch.tensor(
                theta_x.shape[-1], dtype=torch.float, device=pairwise_weight.device
            ) ** torch.tensor(0.5, device=pairwise_weight.device)
        pairwise_weight /= torch.tensor(self.temperature, device=pairwise_weight.device)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):
        # x: [N, C, H, W]
        n = x.size(0)

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == "gaussian":
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == "concatenation":
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        theta_x -= theta_x.mean(dim=-2, keepdim=True)
        phi_x -= phi_x.mean(dim=-1, keepdim=True)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = (
            y.permute(0, 2, 1)
            .contiguous()
            .reshape(n, self.inter_channels, *x.size()[2:])
        )

        # unary_mask: [N, 1, HxW]
        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)
        unary_x = torch.matmul(unary_mask, g_x)
        # unary_x: [N, C, 1, 1]
        unary_x = (
            unary_x.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, 1, 1)
        )

        output = x + self.conv_out(y + unary_x)
        return output


class SEMwithReduction(SEM):
    # +spatial reduction
    def __init__(self, *arg, temperature=0.05, **kwargs):
        super().__init__(*arg, temperature=temperature, **kwargs)
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_mask = nn.Sequential(self.conv_mask, max_pool_layer)


class CADiff(nn.Module):
    def __init__(self, channels=256, reduction=4):
        super(CADiff, self).__init__()
        inter_channels = int(channels // reduction)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )  # SE
        self.sigmoid = nn.Sigmoid()
        self.c3d = conv_t_fusion(channels, channels, 3)

    def forward(self, x, y):
        xa = self.c3d(x, y)
        attn_local = self.local_att(xa)  # (B,C,H,W)
        attn_global = self.global_att(xa)  # (B,C,1,1)
        attn = attn_local + attn_global
        weight = self.sigmoid(attn)  # (B,C,H,W)
        xo = torch.abs(x - y) * (1 + weight)
        return xo


class conv_t_fusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(2, kernel_size, kernel_size),
            stride=(1, 1, 1),
            padding=(0, kernel_size // 2, kernel_size // 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        out = torch.concat((x1, x2), dim=2)
        out = self.act(self.bn(self.conv3d(out).squeeze(2)))
        return out


class MTCAN(nn.Module):
    def __init__(self, num_classes=7, fam_mode="s234d8"):
        super().__init__()
        self.backbone = MSCAN()
        backbone_checkpoint = torch.load(
            "/disk527/Datadisk/b527_syf/checkpoints/mscan_t.pth",
            map_location=torch.device("cpu"),
        )
        new_state_dict = OrderedDict()
        for k, v in backbone_checkpoint["state_dict"].items():
            if k == "head.weight":
                continue
            if k == "head.bias":
                continue
            new_state_dict[k] = v
        self.backbone.load_state_dict(new_state_dict)

        # fam
        self.fam_mode = fam_mode
        if fam_mode == "s234d8":
            self.fusion_ss = nn.Sequential(
                conv3x3(480, 256), nn.BatchNorm2d(256), nn.ReLU()
            )
        elif fam_mode == "s1234d4":
            self.fusion_ss = nn.Sequential(
                conv3x3(512, 256), nn.BatchNorm2d(256), nn.ReLU()
            )
        else:
            raise NotImplementedError()

        # ss
        self.ss_decoder1 = SEMwithReduction(
            in_channels=256,
            reduction=2,
            use_scale=True,
            mode="embedded_gaussian",
            sub_sample=True,
        )
        self.ss_head1 = nn.Sequential(
            conv3x3(256, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

        # bcd
        self.bcd_t_fusion = CADiff(256, 4)
        self.bcd_res = self._make_layer(ResBlock, 256, 128, 2)
        self.bcd_decoder = SEM(
            in_channels=128,
            reduction=2,
            use_scale=True,
            mode="embedded_gaussian",
            temperature=0.05,
        )
        self.bcd_head = nn.Sequential(
            conv1x1(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def FAM(self, x1_feature_list, x2_feature_list):
        if self.fam_mode == "s234d8":
            stage2_shape = x1_feature_list[1].shape[2:]
            f1_234 = torch.concat(
                (
                    x1_feature_list[1],
                    F.interpolate(x1_feature_list[2], stage2_shape, mode="bilinear"),
                    F.interpolate(x1_feature_list[3], stage2_shape, mode="bilinear"),
                ),
                dim=1,
            )
            f1 = self.fusion_ss(f1_234)  # (/8,256)
            f2_234 = torch.concat(
                (
                    x2_feature_list[1],
                    F.interpolate(x2_feature_list[2], stage2_shape, mode="bilinear"),
                    F.interpolate(x2_feature_list[3], stage2_shape, mode="bilinear"),
                ),
                dim=1,
            )
            f2 = self.fusion_ss(f2_234)  # (/8,256)
        elif self.fam_mode == "s1234d4":
            stage1_shape = x1_feature_list[0].shape[2:]
            f1_1234 = torch.concat(
                (
                    x1_feature_list[0],
                    F.interpolate(x1_feature_list[1], stage1_shape, mode="bilinear"),
                    F.interpolate(x1_feature_list[2], stage1_shape, mode="bilinear"),
                    F.interpolate(x1_feature_list[3], stage1_shape, mode="bilinear"),
                ),
                dim=1,
            )
            f1 = self.fusion_ss(f1_1234)  # (/4,256)
            f2_1234 = torch.concat(
                (
                    x2_feature_list[0],
                    F.interpolate(x2_feature_list[1], stage1_shape, mode="bilinear"),
                    F.interpolate(x2_feature_list[2], stage1_shape, mode="bilinear"),
                    F.interpolate(x2_feature_list[3], stage1_shape, mode="bilinear"),
                ),
                dim=1,
            )
            f2 = self.fusion_ss(f2_1234)  # (/4,256)
        return f1, f2

    def SS_forward(self, f1, f2):
        seg1 = self.ss_decoder1(f1)
        seg1 = self.ss_head1(seg1)
        seg2 = self.ss_decoder1(f2)
        seg2 = self.ss_head1(seg2)
        return seg1, seg2

    def BCD_forward(self, f1, f2):
        out = self.bcd_t_fusion(f1, f2)
        out = self.bcd_res(out)
        out = self.bcd_decoder(out)
        out = self.bcd_head(out)
        return out

    def forward(self, x1, x2):
        shape = x1.shape[2:]
        # encoder
        x1_feature_list = self.backbone(x1)
        x2_feature_list = self.backbone(x2)

        # ss branch
        f1_ss, f2_ss = self.FAM(x1_feature_list, x2_feature_list)
        seg1, seg2 = self.SS_forward(f1_ss, f2_ss)

        # bcd branch
        change = self.BCD_forward(f1_ss, f2_ss)
        return (
            F.interpolate(change, shape, mode="bilinear"),
            F.interpolate(seg1, shape, mode="bilinear"),
            F.interpolate(seg2, shape, mode="bilinear"),
        )
