# -*- coding: utf-8 -*-
# @Author           : Sora
# @Contact          : nsoraki@outlook.com
# @File             : hrnet.py
# @Time Create      : 2021/7/26 23:56
# @Contributor      : Sora
import torch
from timm.models.hrnet import HighResolutionNetFeatures, cfg_cls
from torch.nn import functional as F


class HRNet(HighResolutionNetFeatures):

    def forward(self, x):
        _, x1, x2, x3, x4 = super().forward(x)
        _, _, h, w = x1.shape
        x2, x3, x4 = map(lambda t: F.interpolate(t, (h, w), mode='bilinear', align_corners=False), (x2, x3, x4))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


if __name__ == '__main__':
    net = HRNet(cfg=cfg_cls['hrnet_w18_small'], in_chans=3)
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    print(y.shape)
