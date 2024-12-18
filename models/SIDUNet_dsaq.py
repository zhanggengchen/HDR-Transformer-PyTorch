import torch
from torch import nn
from torch.nn import functional as F
from .dsaq_utils.dsaq import ActOurs, Conv2dOurs, ConvTranspose2dOurs

class SpatialAttentionModule(nn.Module):

    def __init__(self, dim, nbits_w, nbits_a):
        super(SpatialAttentionModule, self).__init__()
        self.quant_att1 = ActOurs(nbits_a=nbits_a)
        self.quant_att2 = ActOurs(nbits_a=nbits_a)
        self.att1 = Conv2dOurs(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)
        self.att2 = Conv2dOurs(dim * 2, dim, kernel_size=3, padding=1, bias=True, nbits_w=nbits_w, nbits_a=nbits_a)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.quant_att1(x1)
        x2 = self.quant_att2(x2)
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class SIDUNet(nn.Module):
    def __init__(self, inchannels=6, outchannels=3, channels=64, nbits_w=4, nbits_a=4) -> None:
        super().__init__()

        # coarse feature
        self.conv_f1 = nn.Conv2d(inchannels, channels, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(inchannels, channels, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(inchannels, channels, 3, 1, 1)
        # spatial attention module
        self.att_module_l = SpatialAttentionModule(channels, nbits_w=nbits_w, nbits_a=nbits_a)
        self.att_module_h = SpatialAttentionModule(channels, nbits_w=nbits_w, nbits_a=nbits_a)
        self.quant_f1_att = ActOurs(nbits_a=nbits_a)
        self.quant_f2 = ActOurs(nbits_a=nbits_a)
        self.quant_f3_att = ActOurs(nbits_a=nbits_a)
        self.conv_first = Conv2dOurs(channels * 3, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)

        self.conv1_1 = Conv2dOurs(channels, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv1_2 = Conv2dOurs(channels, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = Conv2dOurs(channels, channels * 2, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv2_2 = Conv2dOurs(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = Conv2dOurs(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv3_2 = Conv2dOurs(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = Conv2dOurs(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv4_2 = Conv2dOurs(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = Conv2dOurs(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv5_2 = Conv2dOurs(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)

        self.upv6 = ConvTranspose2dOurs(channels * 16, channels * 8, 2, stride=2, nbits_w=nbits_w, nbits_a=nbits_a)
        self.quant_upv6 = ActOurs(nbits_a=nbits_a)
        self.quant_conv4 = ActOurs(nbits_a=nbits_a)
        self.conv6_1 = Conv2dOurs(channels * 16, channels * 8, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)
        self.conv6_2 = Conv2dOurs(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)

        self.upv7 = ConvTranspose2dOurs(channels * 8, channels * 4, 2, stride=2, nbits_w=nbits_w, nbits_a=nbits_a)
        self.quant_upv7 = ActOurs(nbits_a=nbits_a)
        self.quant_conv3 = ActOurs(nbits_a=nbits_a)
        self.conv7_1 = Conv2dOurs(channels * 8, channels * 4, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)
        self.conv7_2 = Conv2dOurs(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)

        self.upv8 = ConvTranspose2dOurs(channels * 4, channels * 2, 2, stride=2, nbits_w=nbits_w, nbits_a=nbits_a)
        self.quant_upv8 = ActOurs(nbits_a=nbits_a)
        self.quant_conv2 = ActOurs(nbits_a=nbits_a)
        self.conv8_1 = Conv2dOurs(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)
        self.conv8_2 = Conv2dOurs(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)

        self.upv9 = ConvTranspose2dOurs(channels * 2, channels, 2, stride=2, nbits_w=nbits_w, nbits_a=nbits_a)
        self.quant_upv9 = ActOurs(nbits_a=nbits_a)
        self.quant_conv1 = ActOurs(nbits_a=nbits_a)
        self.conv9_1 = Conv2dOurs(channels * 2, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, quant_act=False)
        self.conv9_2 = Conv2dOurs(channels, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)

        self.conv_after_body = Conv2dOurs(channels, channels, kernel_size=3, stride=1, padding=1, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv_last = nn.Conv2d(channels, outchannels, 3, 1, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2, x3):
        # coarse feature
        f1 = self.conv_f1(x1)
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)

        # spatial feature attention 
        f1_att_m = self.att_module_h(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_l(f3, f2)
        f3_att = f3 * f3_att_m
        f1_att = self.quant_f1_att(f1_att)
        q_f2 = self.quant_f2(f2)
        f3_att = self.quant_f3_att(f3_att)
        x = self.conv_first(torch.cat((f1_att, q_f2, f3_att), dim=1))

        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = self.quant_upv6(up6)
        conv4 = self.quant_conv4(conv4)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = self.quant_upv7(up7)
        conv3 = self.quant_conv3(conv3)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = self.quant_upv8(up8)
        conv2 = self.quant_conv2(conv2)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = self.quant_upv9(up9)
        conv1 = self.quant_conv1(conv1)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))

        res = self.conv_after_body(conv9 + x)
        out = self.conv_last(f2 + res)
        out = torch.sigmoid(out)

        return out
    
    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = 16

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor
        
    def _check_and_crop(self, x):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top:bottom, left:right]
        return x