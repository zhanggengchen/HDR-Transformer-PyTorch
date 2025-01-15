#-*- coding:utf-8 -*-  
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)
    

class L1Mu1Loss(nn.Module):
    def __init__(self, alpha=0.1, mu=5000):
        super(L1Mu1Loss, self).__init__()
        self.mu = mu
        self.alpha = alpha

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label) + self.alpha * nn.L1Loss()(pred, label)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss


class JointReconPerceptualPsnrlLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualPsnrlLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1Mu1Loss(mu=self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss
    
class TernaryLoss(nn.Module):
    def __init__(self, patch_size=7):
        super(TernaryLoss, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to('cuda')

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
        
    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss
    
class SAFLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.1, mu=5000):
        super(SAFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)
        self.census_loss = TernaryLoss()
    
    def forward(self, pred_m, pred, target):
        pred_mu = range_compressor(pred, self.mu)
        pred_m_mu = range_compressor(pred_m, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(pred, target)
        loss_vgg = self.loss_vgg(pred_mu, target_mu)
        loss_r = loss_recon + self.alpha * loss_vgg
        # loss_m = self.loss_recon(pred_m, target) + self.census_loss(pred_m_mu, target_mu)
        loss_m = self.loss_recon(pred_m, target)
        loss = loss_r + self.beta * loss_m
        loss = loss_r
        return loss