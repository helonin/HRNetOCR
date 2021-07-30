# -*- coding: utf-8 -*- 
# @Author           : Sora
# @Contact          : nsoraki@outlook.com
# @File             : segmentor.py 
# @Time Create      : 2021/7/28 9:16 
# @Contributor      : Sora
from torch.nn import *
from .hrnet import HRNet, cfg_cls
from .ocr import OCRNet


class HRNetOCR(Sequential):

    def __init__(self, num_classes: int, in_channels=3, cfg=cfg_cls['hrnet_w18_small']):
        super(HRNetOCR, self).__init__(
            HRNet(in_chans=in_channels, cfg=cfg),
            OCRNet(channels=128 + 256 + 512 + 1024, num_classes=num_classes)
        )
