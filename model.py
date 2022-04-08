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
import segmentation_models_pytorch as smp

SMP_ENCODER = 'resnet18'
SMP_EN_WEIGHTS='imagenet'


class Correlation_Transformer(nn.Module):
    def __init__(self, in_channels) :
        super(Correlation_Transformer, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,5), stride=(1,2), padding=(1,2))
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, input, target):
        # inputs: [B,S,C,H,W]: inputs need to transformer to align to target
        # target:[B,C,H,W]
        B,S,C,H,W = input.size()

        target_f = target.permute(0,1,3,2).flatten(1,2) #[B,C',H]
        input_f = input.permute(0,1,2,4,3).flatten(2,3) #[B,S,C',H]

        target_f = F.normalize(target_f, dim=1)
        input_f = F.normalize(input_f, dim=2)

        # # conv to [B,C',H,W']
        # features = torch.cat((target,input.flatten(0,1)),dim=0) #target+input [B+B*S,C,H,W]
        # features = self.conv1(features) #[B,8,H.W//2]
        # features = self.relu(features)
        # features = self.conv2(features) #[B,16,H.W//4]
        # features = self.relu(features)
        # features = self.conv3(features) #[B,32,H.W//8]
        # features = self.relu(features)
        # features = self.conv4(features) #[B,64,H.W//16]
        # features = self.relu(features)
        # features = features.permute(0,1,3,2).flatten(1,2) #[B,W//16*64,H]
        # features = F.normalize(features,dim=1) #[B,W//16*64,H]
        #
        # target_f = features[:B] #[B,C',H]
        # input_f = features[B:].view(B,S,-1,H) #[B,S,C',H]

        correlation_map = torch.tensor([],device=input_f.device)
        for i in range(B):
            correlation = F.conv1d(input_f[i], target_f[i].unsqueeze(0), padding=(H-1)) # S,1,2H-1
            correlation = correlation[:,0,:H]/torch.range(1,H,device=input_f.device) # S, H
            correlation_map = torch.cat((correlation_map, correlation.unsqueeze(0)), dim=0)  # B,S,H

        overlap = torch.argmax(correlation_map,dim=1) #B,S
        shift = H-1-overlap #B,S
        out = torch.zeros_like(input)
        out[torch.arange(B),torch.arange(S),:,:overlap,:] = input[torch.arange(B),torch.arange(S),:,shift:,:] #??

        #kernel = torch.softmax(correlation_map,dim=-1)
        # # conv to transform the original inputs
        # input = input.permute(0,1,2,4,3).reshape(B*S,C*W,1,H) #[B*S,C*W,1,H]
        # kernel = kernel.flatten(0, 1)[:,None,None,:] #[B*S,1,1,H]
        #
        # if 0:#debug
        #     kernel = torch.zeros_like(kernel)
        #     kernel[:,:,:,H//2-3] = 0.8
        #     kernel[:, :, :, H // 2 - 4] = 0.2
        # out = torch.tensor([],device=input_f.device)
        # for i in range(B*S):
        #     out_f = F.conv1d(input[i], kernel[i], padding=(H-1)) # C*W,1,2H-1
        #     out_f = out_f[:,0,:H] # C*W, H
        #     out = torch.cat((out, out_f), dim=0)  # B*S,C*W,H
        # out = out.view(B,S,C,W,H).permute(0,1,2,4,3)
        #
        if 1:#debug
            img = transforms.functional.to_pil_image(out[0,0],mode='RGB')
            img.save('shift.png')

        return out

class Net(nn.Module):
    def __init__(self, sequence=16):
        super(Net, self).__init__()
        self.sequence = sequence

        #self.shift_align = Correlation_Transformer(3)
        self.reflec_detect = smp.Unet(encoder_name=SMP_ENCODER,encoder_weights=SMP_EN_WEIGHTS,in_channels=sequence*3, classes=1)#,activation=SMP_ACTIVATION )
    def forward(self, left, light):
        # left,right from current to previous #[b,s,c,h,w]
        # debug, save inputs
        if 0:#debug
            for i in range(left.size(1)):
                img = transforms.functional.to_pil_image(left[0,i],mode='RGB')
                img.save('input_left_'+str(i)+'.png')
            img = transforms.functional.to_pil_image(light[0, -1], mode='RGB')
            img.save('input_right.png')

        # # align the 6DOF or 5DOF
        # aligned_f = self.align(x[:,:-1], x[:,-1])
        # aligned_f = torch.cat((aligned_f, x[:,-1:]), dim=1)

        # only left
        prd_mask = self.reflec_detect(left.flatten(1,2))

        # both left and right
        # both = torch.cat([left,right],dim=1) #[b,2s,c,h,w]
        # prd_mask = self.reflec_detect(both.flatten(1, 2))

        prd_mask = torch.sigmoid(prd_mask.squeeze(1))

        if 0:# debug
            img = transforms.functional.to_pil_image((prd_mask[0]>=0.5).float(),mode='L')
            #img = transforms.functional.to_pil_image(prd_mask[0], mode='L')
            img.save('predict_mask.png')

        return prd_mask
    