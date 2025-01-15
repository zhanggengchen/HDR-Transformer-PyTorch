# -*- coding:utf-8 -*-
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from models.SCTNet import SCTNet
from models.SAFNet_S import SAFNet_S
from models.SAFNet_M import SAFNet_M
from models.SAFNet import SAFNet
from models.SIDUNet import SIDUNet
from ptflops import get_model_complexity_info


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 256 #(128 // upscale // window_size + 1) * window_size
    width = 256 #(128 // upscale // window_size + 1) * window_size

    # model = SCTNet(upscale=2, img_size=(height, width), in_chans=18,
    #               window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
    #               embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)

    # model = SCTNet(upscale=2, img_size=(height, width), in_chans=18,
    #             window_size=window_size, img_range=1., depths=[3, 3, 3, 3],
    #             embed_dim=36, num_heads=[6, 6, 6, 6], mlp_ratio=2)

    model = SIDUNet()

    # model = SAFNet_S()

    # model = SAFNet_M()

    macs, params = get_model_complexity_info(model, (6, 256, 256), as_strings=True, backend='pytorch',
                                           print_per_layer_stat=True, verbose=True)

    # macs, params = get_model_complexity_info(model, (18, 256, 256), as_strings=True, backend='pytorch',
    #                                        print_per_layer_stat=True, verbose=True)
    
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
