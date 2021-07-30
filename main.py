# -*- coding: utf-8 -*- 
# @Author           : Sora
# @Contact          : nsoraki@outlook.com
# @File             : main.py 
# @Time Create      : 2021/7/28 9:22 
# @Contributor      : Sora
import torch

from models import HRNetOCR

if __name__ == '__main__':
    net = HRNetOCR(num_classes=2)
    x = torch.rand(1, 3, 224, 224)
    y = net(x)

    print(y[0].shape, y[1].shape)
