# -*- coding:utf-8 -*-
import os
import os.path as osp
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset, SIG17_Test_Dataset
from models.loss import L1MuLoss, JointReconPerceptualLoss, JointReconPerceptualPsnrlLoss
from models.SIDUNet_dsaq import SIDUNet
from utils.utils import *
from utils.training import *
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory'),
    parser.add_argument('--patch_size', type=int, default=256),
    parser.add_argument("--sub_set", type=str, default='sig17_training_crop128_stride64',
                        help='dataset directory')
    #Experiment path
    parser.add_argument('--logdir', type=str, default='./experiment/SIDUNet_dsaq_3b',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    # Model
    parser.add_argument('--nbits_w', type=int, default=4,
                        help='quantization bits for weight')
    parser.add_argument('--nbits_a', type=int, default=4,
                        help='quantization bits for activation')
    # Training
    parser.add_argument('--resume', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=1,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)') # learning rate 减少一半
    parser.add_argument('--lr_decay_interval', type=int, default=100,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=60, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()

def main():
    # settings
    args = get_args()
    # random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # logger
    logger = infoLogger("trainLogger", osp.join(args.logdir, "train.log"))
    tbWriter = SummaryWriter(os.path.join(args.logdir, 'tensorboard'))
    # model architectures
    model = SIDUNet(inchannels=6, outchannels=3, channels=64, nbits_w=args.nbits_w, nbits_a=args.nbits_a)
    cur_psnr = [-1.0]
    # init
    if args.init_weights:
        init_parameters(model)
    # loss
    loss_dict = {
        0: L1MuLoss,
        1: JointReconPerceptualLoss,
        2: JointReconPerceptualPsnrlLoss,
        }
    criterion = loss_dict[args.loss_func]().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
        else:
            logger.info("===> No checkpoint is founded at {}.".format(args.resume))
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logger.info("===> Loading checkpoint from: {}".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("===> Loaded pretrained model {}".format(args.pretrained))
        else:
            logger.info("===> No pretrained model is founded at {}.".format(args.pretrained))
    
    # dataset and dataloader
    train_dataset = SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=True, crop_size=512)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dataset_size = len(train_loader.dataset)
    logger.info(f'''===> Start training DSAQ-UNet

        Dataset dir:     {args.dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.lr}
        Training size:   {dataset_size}
        Device:          {device.type}
        Weight Bits:     {args.nbits_w}
        Activation Bits: {args.nbits_a}
        ''')

    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion, logger)
        # validation(args, model, device, val_loader, optimizer, epoch, criterion, cur_psnr)
        test(args, model, device, optimizer, epoch, cur_psnr, logger, tbWriter)


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=2,3 python train_dsaq_unet.py --loss_func 2 --logdir ./experiment/SIDUNet_dsaq_3b_loss2 --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 3 --nbits_a 3
# CUDA_VISIBLE_DEVICES=0,1 python train_dsaq_unet.py --loss_func 2 --logdir ./experiment/SIDUNet_dsaq_2b_loss2 --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 2 --nbits_a 2
# CUDA_VISIBLE_DEVICES=0,1 python train_dsaq_unet.py --loss_func 2 --logdir ./experiment/SIDUNet_dsaq_4b_loss2 --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 4 --nbits_a 4
# CUDA_VISIBLE_DEVICES=0,1 python train_dsaq_unet.py --loss_func 2 --logdir ./experiment/SIDUNet_dsaq_8b_loss2 --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 8 --nbits_a 8
# CUDA_VISIBLE_DEVICES=0,1 python train_dsaq_unet.py --logdir ./experiment/SIDUNet_dsaq_8b --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 8 --nbits_a 8
# CUDA_VISIBLE_DEVICES=2,3 python train_dsaq_unet.py --lr 0.00005 --loss_func 2 --logdir ./experiment/SIDUNet_dsaq_4b_loss2_lr05 --pretrained experiment/SIDUNet/best_checkpoint.pth --nbits_w 4 --nbits_a 4