# -*- coding:utf-8 -*-
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from models.SCTNet import SCTNet


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 256 #(128 // upscale // window_size + 1) * window_size
    width = 256 #(128 // upscale // window_size + 1) * window_size

    model = SCTNet(upscale=2, img_size=(height, width), in_chans=18,
                window_size=window_size, img_range=1., depths=[3, 3, 3, 3],
                embed_dim=36, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 6, height, width))
    x = model(x, x, x)
    print(x.shape)
