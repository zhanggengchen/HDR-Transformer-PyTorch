#-*- coding:utf-8 -*-  
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse

import cv2
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

from dataset.dataset_sig17 import SIG17_Test_Dataset
from models.hdr_transformer import HDRTransformer
from models.SCTNet import SCTNet
# from models.SIDUNet import SIDUNet
# from models.SIDUNet_lsq import SIDUNet
from models.SIDUNet_dsaq import SIDUNet
from models.SwinIR import SwinIR
from train import test_single_img
from utils.utils import *

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/hdr_transformer_ckpt.pth')
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer")
parser.add_argument('--model_arch', type=int, default=0)
parser.add_argument('--nbits_w', type=int, default=4)
parser.add_argument('--nbits_a', type=int, default=4)


def main():
    # Settings
    args = parser.parse_args()

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = infoLogger("testLogger", osp.join(args.save_dir, "test.log"))

    # pretrained_model
    logger.info(">>>>>>>>> Start Testing >>>>>>>>>")
    logger.info(f"Load weights from: {args.pretrained_model}")

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # model architecture
    model_dict = {
        0: HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6),
        1: SCTNet(img_size=(72, 72), in_chans=18,
                            window_size=8, img_range=1., depths=[6, 6, 6, 6],
                            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect'),
        # 2: SIDUNet(inchannels=6, outchannels=3, channels=64),
        2: SIDUNet(inchannels=6, outchannels=3, channels=64, nbits_w=args.nbits_w, nbits_a=args.nbits_a),
        3: SwinIR(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6),
        4: SCTNet(upscale=2, img_size=(72, 72), in_chans=18, window_size=8, img_range=1., depths=[3, 3, 3, 3],
            embed_dim=36, num_heads=[6, 6, 6, 6], mlp_ratio=2),
    }
    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
    model.eval()

    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    for idx, img_dataset in enumerate(datasets):
        pred_img, label = test_single_img(model, img_dataset, device)
        pred_hdr = pred_img.copy()
        pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]
        # psnr-l and psnr-\mu
        scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)
        scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
        # ssim-l
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img, label)
        # ssim-\mu
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)

        logger.info("Image_{}, PSNR_mu/SSIM_mu: {:.4f}/{:.4f}, PSNR_l/SSIM_l: {:.4f}/{:.4f}".format(idx, scene_psnr_mu, scene_ssim_mu, scene_psnr_l, scene_ssim_l))

        # save results
        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            cv2.imwrite(os.path.join(args.save_dir, '{}_pred.png'.format(idx)), pred_img_mu)
            save_hdr(os.path.join(args.save_dir, '{}_pred.hdr'.format(idx)), pred_hdr)

    logger.info("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    logger.info("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    logger.info(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=0 python test.py --pretrained_model experiment/SIDUNet_dsaq_4b/best_checkpoint.pth --save_dir ./results/SIDUNet_dsaq_4b --model_arch 2 --nbits_w 4 --nbits_a 4
# CUDA_VISIBLE_DEVICES=0 python test.py --pretrained_model experiment/SIDUNet_dsaq_3b_1/best_checkpoint.pth --save_dir ./results/SIDUNet_dsaq_3b_1 --model_arch 2 --nbits_w 3 --nbits_a 3

# CUDA_VISIBLE_DEVICES=0 python test.py --pretrained_model experiment/sctnet_30g/best_checkpoint.pth --save_dir ./results/sctnet_30g --model_arch 4