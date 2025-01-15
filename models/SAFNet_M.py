import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import warp, merge_hdr
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


div_size = 16
div_flow = 20.0


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(6, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        
    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class RefineNet(nn.Module):
    def __init__(self, embed_dim=36):
        super(RefineNet, self).__init__()
        # self.conv0 = nn.Sequential(convrelu(6, 20), convrelu(20, 11))
        # self.conv1 = nn.Sequential(convrelu(6+2+2+1+1+3, 40), convrelu(40, 20))
        # self.conv2 = nn.Sequential(convrelu(6, 20), convrelu(20, 11))

        self.conv1 = nn.Conv2d(6, embed_dim//3, 3, 1, 1)
        self.conv2 = nn.Conv2d(12, embed_dim//3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, embed_dim//3, 3, 1, 1)

        self.conv_end = nn.Conv2d(embed_dim//3, embed_dim, 3, 1, 1)
        self.conv_first = Mlp(embed_dim, embed_dim, act_layer=nn.GELU)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.resblock1 = ResBlock(42, 1)
        self.resblock2 = ResBlock(42, 2)
        self.resblock3 = ResBlock(42, 4)
        self.resblock4 = ResBlock(42, 2)
        self.resblock5 = ResBlock(42, 1)
        self.conv3 = nn.Conv2d(42, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x1, x2, x3 = x.chunk(3, dim=2)
        x = self.conv_first(x)
        for layer, layer_lm, layer_rm, mlp in zip(self.layers, self.channel_layers_lm, self.channel_layers_rm, self.mlps):
            x = layer(x, x_size)
            x11, x22, x33 = x.chunk(3, dim=2)
            #print("x2 {}".format(x2.shape))
       	    #print("x22 {}".format(x22.shape))

            x_lm = layer_lm(x2+x22, x1+x11, x_size)
            x_rm = layer_rm(x2+x22, x3+x33, x_size)
            fuse = torch.cat((x_lm, x2 + x22, x_rm), dim=2)
            x = mlp(fuse)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x
        
    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([img1_c, flow0 / div_flow, flow2 / div_flow, mask0, mask2, img_hdr_m], 1))
        feat2 = self.conv2(img2_c)
        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        res = self.conv3(feat)
        img_hdr_r = torch.clamp(img_hdr_m + res, 0, 1)
        return img_hdr_r


class SAFNet_M(nn.Module):
    def __init__(self):
        super(SAFNet_M, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (int(div_size * np.ceil(h * scale_factor / div_size)), int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5)
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4)
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3)
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(up_mask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_mask2_1 = F.interpolate(up_mask2_1, size=org_size, mode='bilinear', align_corners=False)

        up_mask0_1 = torch.sigmoid(up_mask0_1)
        up_mask2_1 = torch.sigmoid(up_mask2_1)

        return up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1
    
    # def forward(self, img0_c, img1_c, img2_c, scale_factor=0.5, refine=True, training=True):
    #     # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
    #     flow0, flow2, mask0, mask2 = self.forward_flow_mask(img0_c, img1_c, img2_c, scale_factor=scale_factor)

    #     img0_c_warp = warp(img0_c, flow0)
    #     img2_c_warp = warp(img2_c, flow2)
    #     img_hdr_m = merge_hdr(
    #         [img0_c_warp[:, 3:6, :, :], img1_c[:, 3:6, :, :], img2_c_warp[:, 3:6, :, :]], 
    #         [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
    #         mask0, mask2
    #         )
        
    #     if refine == True:
    #         img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m)
    #         return img_hdr_m, img_hdr_r
    #     else:
    #         return img_hdr_m
        
    def forward(self, img0_c, scale_factor=0.5, refine=True):
        img1_c = img0_c
        img2_c = img0_c
        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        flow0, flow2, mask0, mask2 = self.forward_flow_mask(img0_c, img1_c, img2_c, scale_factor=scale_factor)

        img0_c_warp = warp(img0_c, flow0)
        img2_c_warp = warp(img2_c, flow2)
        img_hdr_m = merge_hdr(
            [img0_c_warp[:, 3:6, :, :], img1_c[:, 3:6, :, :], img2_c_warp[:, 3:6, :, :]], 
            [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
            mask0, mask2
            )
        
        if refine == True:
            img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m)
            # return img_hdr_m, img_hdr_r
            return img_hdr_r
        else:
            return img_hdr_m