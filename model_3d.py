#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 2022

@author: shan
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch=64, kernel_size=3, active=True):
        super(Conv3DBlock, self).__init__()
        self.active = active
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size, padding='same', bias=False)
        self.norm = nn.GroupNorm(32, out_ch) # 64c to 32 group
    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x)
        if self.active:
            x = torch.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv3d_1 = Conv3DBlock(in_ch, out_ch, kernel_size=1, active=False)
        self.conv3d_2 = Conv3DBlock(out_ch, out_ch, kernel_size=3)
        self.conv3d_3 = Conv3DBlock(out_ch, out_ch, kernel_size=3)
        self.norm = nn.GroupNorm(out_ch//2,out_ch) # 64c to 32 group
    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.norm(x)
        return x

class ResPath(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResPath, self).__init__()
        self.conv3d_1 = Conv3DBlock(in_ch, out_ch, kernel_size=1, active=False)
        self.conv3d_3 = Conv3DBlock(in_ch, out_ch, kernel_size=3)
        self.norm = nn.GroupNorm(out_ch//2, out_ch) # 64c to 32 group
    def forward(self, x, length):
        shortcut = self.conv3d_1(x)
        x = self.conv3d_3(x)
        x = shortcut+x
        x = torch.relu(x)
        x = self.norm(x)
        for i in range(length-1):
            shortcut = self.conv3d_1(x)
            x = self.conv3d_3(x)
            x = torch.relu(x)
            x = shortcut + x
            x = torch.relu(x)
            x = self.norm(x)
        return x

class Conv3DNet(nn.Module):
    def __init__(self, sequence=16, out_ch=1):
        super(Conv3DNet, self).__init__()
        self.sequence = sequence
        self.res_block_1 = ResBlock(3, 64)
        self.res_block_2 = ResBlock(64, 64)
        self.res_path_1 = ResPath(64, 64)
        self.res_path_2 = ResPath(64, 128)
        self.conv3d_block_1 = Conv3DBlock(3, 64, kernel_size=3)
        self.conv3d_block_2 = Conv3DBlock(64, 64, kernel_size=3)
        self.conv3d_block_3 = Conv3DBlock(64, 128, kernel_size=3)
        self.norm = nn.GroupNorm(32, 64)
        # level 3
        self.avgpool3d_1 = nn.AvgPool3d((sequence//8,1,1), stride=1)
        self.conv2d_1 = nn.Conv2d(64, 2048, 7, padding='same')
        self.conv2d_2 = nn.Conv2d(2048, 2048, 1, padding='same')
        self.conv2d_3 = nn.Conv2d(2048, out_ch, 1, padding='same')
        self.dropout = nn.Dropout(p = 0.5)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2, bias=False)
        # level 2
        self.avgpool3d_2 = nn.AvgPool3d((sequence//4,1,1), stride=1)
        self.conv2d_4 = nn.Conv2d(128, out_ch, 1, padding='same')
        self.conv2d_5 = nn.Conv2d(out_ch*2, out_ch, 1, padding='same')
        # level 1
        self.avgpool3d_3 = nn.AvgPool3d((sequence//2,1,1), stride=1)
        self.conv2d_6 = nn.Conv2d(64, out_ch, 1, padding='same')


    def forward(self, left, light):
        # left,right from current to previous #[b,s,c,h,w]
        # debug, save inputs
        if 0:#debug
            for i in range(left.size(1)):
                img = transforms.functional.to_pil_image(left[0,i],mode='RGB')
                img.save('input_left_'+str(i)+'.png')
            img = transforms.functional.to_pil_image(light[0, -1], mode='RGB')
            img.save('input_right.png')

        # Block1
        left = left.permute(0,2,1,3,4) #[b,s,3,h,w->b,c,s,h,w]
        part1,part2 = torch.split(left, self.sequence//2, dim=2) #[b,3,s/2,h,w] # split on sequences??
        part1 = self.res_block_1(part1) #[b,64,s//2,h,w]
        part2 = self.conv3d_block_1(part2) #[b,64,s//2,h,w]
        left = torch.cat([part1,part2],dim=2) #[b,64,s,h,w]
        left = self.norm(left)
        left = torch.max_pool3d(left,2) # [b,64,s//2,h//2,w//2]

        part1, part2 = torch.split(left, left.size(2)//2, dim=2) # [b,64,s//4,h//2,w//2]
        part1_f = self.res_path_1(part1, 2) # [b,64,s//4,h//2,w//2]
        part2_f = self.conv3d_block_2(part2) # [b,64,s//4,h//2,w//2]
        f1 = torch.cat([part1_f,part2_f],dim=2) # [b,64,s//2,h//2,w//2]

        # Block 2 # only res_block on part1
        part1 = self.res_block_2(part1) # [b,64,s//4,h//2,w//2]
        left = torch.cat([part1, part2], dim=2) # [b,64,s//2,h//2,w//2]
        left = self.norm(left)
        left = torch.max_pool3d(left,2) # [b,64,s//4,h//4,w//4]

        part1, part2 = torch.split(left, left.size(2)//2, dim=2) # [b,64,s//8,h//4,w//4]
        part1_f = self.res_path_2(part1, 1) # [b,128,s//8,h//4,w//4]
        part2_f = self.conv3d_block_3(part2) # [b,128,s//8,h//4,w//4]
        f2 = torch.cat([part1_f,part2_f],dim=2) # [b,128,s//4,h//4,w//4]

        # Block 3 # only res_block on part1
        part1 = self.res_block_2(part1) # [b,64,s//8,h//4,w//4]
        left = torch.cat([part1, part2], dim=2) # [b,64,s//4,h//4,w//4]
        left = self.norm(left)
        left = torch.max_pool3d(left, 2) # [b,64,s//8,h//8,w//8]

        # remove s
        if left.size(2) > 1:
            f3 = self.avgpool3d_1(left) ##[b,64,1,h//8,w//8]
        f3 = f3.squeeze(2) #[b,64,h//8,w//8]

        # extract feature
        f3 = self.conv2d_1(f3) #[b,2048,h//8,w//8]
        f3 = torch.relu(f3)
        f3 = self.dropout(f3)
        f3 = self.conv2d_2(f3) #[b,2048,h//8,w//8]
        f3 = torch.relu(f3)
        f3 = self.dropout(f3)
        # predict layer
        f3 = self.conv2d_3(f3) #[b,out_ch,h//8,w//8]
        # deconv
        f3 = self.deconv(f3) #[b,out_ch,h//4,w//4]

        # features from f2 layer
        f2 = self.avgpool3d_2(f2) #[b,128,1,h//4,w//4]
        f2 = f2.squeeze(2)  #[b,128,h//4,w//4]
        # predict layer
        f2 = self.conv2d_4(f2) #[b,out_ch,h//4,w//4]
        f2 = torch.cat([f3,f2], dim=1) #[b,out_ch*2,h//4,w//4]
        f2 = self.conv2d_5(f2) #[b,out_ch,h//4,w//4]
        f2 = self.deconv(f2) #[b,out_ch,h//2,w//2]

        # features from f1 layer
        f1 = self.avgpool3d_3(f1) #[b,64,1,h//2,w//2]
        f1 = f1.squeeze(2)  # [b, 64,h//2,w//2]
        f1 = self.conv2d_6(f1) # [b, out_ch ,h//2,w//2]
        f1 = torch.cat([f2, f1], dim=1) #[b, out_ch*2,h//2,w//2]
        f1 = self.conv2d_5(f1) #[b, out_ch,h//2,w//2]
        f1 = self.deconv(f1) #[b, out_ch,h,w]

        # softmax
        #prd_mask = torch.softmax(f1)
        prd_mask = torch.sigmoid(f1.squeeze(1))

        if 0:# debug
            img = transforms.functional.to_pil_image((prd_mask[0]>=0.5).float(),mode='L')
            img.save('predict_mask.png')

        return prd_mask
    