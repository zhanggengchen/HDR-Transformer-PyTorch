# -*- coding:utf-8 -*-
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from models.SIDUNet_lsq import SIDUNet as QSIDUNet
from models.SIDUNet import SIDUNet
from utils.bops_counter import get_model_complexity_info

if __name__ == '__main__':

    height = 256 #(128 // upscale // window_size + 1) * window_size
    width = 256 #(128 // upscale // window_size + 1) * window_size

    model = QSIDUNet(nbits_w=2, nbits_a=2)
    # model = SIDUNet()

    flops_count, params_count = get_model_complexity_info(model, input_shape=(6, height, width))

    print(flops_count, params_count)

