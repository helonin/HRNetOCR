# -*- coding: utf-8 -*- 
# @Author           : Sora
# @Contact          : nsoraki@outlook.com
# @File             : ocr.py
# @Time Create      : 2021/7/26 23:38
# @Contributor      : Sora

import torch
from torch.nn import *
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from einops import rearrange


class SpatialGather(Module):

    def __init__(self, num_classes: int, scale: float = 1):
        super(SpatialGather, self).__init__()
        self.cls_num = num_classes
        self.scale = scale

    def forward(self, feats, probs):
        probs = rearrange(probs, 'b k h w -> b k (h w)')  # b x k x (h w)
        feats = rearrange(feats, 'b c h w -> b (h w) c')  # b x (h w) x c
        probs = torch.softmax(self.scale * probs, dim=-1)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = rearrange(ocr_context, 'b k c -> b c k 1')
        return ocr_context


class ObjectAttentionBlock2D(Module):

    def __init__(self, in_channels: int, key_channels: int, scale: int = 1, bn_type='BN'):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.pool = MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else Identity()
        self.up = UpsamplingBilinear2d(scale_factor=scale) if scale > 1 else Identity()
        self.to_q = Sequential(
            ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type)),
            ConvModule(key_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        )
        self.to_k = Sequential(
            ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type)),
            ConvModule(key_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        )
        self.to_v = ConvModule(in_channels, key_channels, 1, bias=False, norm_cfg=dict(type=bn_type))
        self.f_up = ConvModule(key_channels, in_channels, 1, bias=False, norm_cfg=dict(type=bn_type))

    def forward(self, feats, context):
        b, c, h, w = feats.shape
        feats = self.pool(feats)

        query = rearrange(self.to_q(feats), 'b c h w -> b (h w) c ')
        key = rearrange(self.to_k(context), 'b c k 1 -> b c k')
        value = rearrange(self.to_v(context), 'b c k 1 -> b k c ')

        sim_map = torch.matmul(query, key)  # b l k
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  # b l c

        context = rearrange(context, 'b (h w) c ->b c h w', h=h, w=w)

        context = self.f_up(context)
        context = self.up(context)

        return context


class SpatialOCR(Module):

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, bn_type='BN'):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, bn_type)
        self.conv_bn_dropout = Sequential(
            ConvModule(2 * in_channels, out_channels, 1, bias=False, norm_cfg=dict(type='BN')),
            Dropout2d(dropout)
        )

    def forward(self, feats, context):
        context = self.object_context_block(feats, context)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class OCRNet(Module):
    def __init__(self, channels, num_classes, ocr_mid_channels=512, ocr_key_channels=256):
        super(OCRNet, self).__init__()

        self.soft_object_regions = Sequential(
            ConvModule(channels, channels, 1),
            Conv2d(channels, num_classes, 1)
        )

        self.pixel_representations = ConvModule(channels, ocr_mid_channels, 3, 1, 1)

        self.object_region_representations = SpatialGather(num_classes)

        self.object_contextual_representations = SpatialOCR(in_channels=ocr_mid_channels,
                                                            key_channels=ocr_key_channels,
                                                            out_channels=ocr_mid_channels,
                                                            scale=1,
                                                            dropout=0.05,
                                                            )
        self.augmented_representation = Conv2d(ocr_mid_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        out_aux = self.soft_object_regions(feats)  # b k h w

        feats = self.pixel_representations(feats)  # b c h w

        context = self.object_region_representations(feats, out_aux)  # b c k 1

        feats = self.object_contextual_representations(feats, context)  # b c h w

        out = self.augmented_representation(feats)  # b k h w

        return out_aux, out


if __name__ == '__main__':
    x = torch.rand(1, 64, 56, 56)
    net = OCRNet(64, 2)
    print(net(x)[0].shape, net(x)[1].shape)
