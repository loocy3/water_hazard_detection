#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:06:23 2021

@author: shan
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

# get partial correlation similarity function
# inputs:
#   grd_feature, sat_feature
# return:
#   similarity_matrix
class partical_similarity(nn.Module):
    def __init__(self, shift_range):
        super(partical_similarity, self).__init__()
        self.shift_range = shift_range
        
    def forward(self, grd_feature, sat_feature, test_method=2):
        # test_method: 0:sqrt, 1:**2, 2:crop
        
        # use grd as kernel, sat map as inputs, convolution to get correlation
        M,C,H_s,W_s = sat_feature.size()
        
        _,_,H,W = grd_feature.size()
        kernel_H = min(H,H_s - 2*self.shift_range,24)
        kernel_W = min(W,W_s - 2*self.shift_range,24)
        
        # only use kernel_edge in center as kernel, for efficient process time
        W_start = W//2-kernel_W//2
        W_end = W//2+kernel_W//2
        H_start = H//2-kernel_H//2
        H_end = H//2+kernel_H//2
        grd_feature = grd_feature[:,:,H_start:H_end,W_start:W_end]
        N,_,H_k,W_k = grd_feature.size()
        if test_method != 2: #not crop_method: # normalize later
            grd_feature = F.normalize(grd_feature.reshape(N,-1)) 
            grd_feature = grd_feature.view(N,C,H_k,W_k)
        
        # only use kernel_edge+2shift_range in center as input
        W_start = W_s//2-self.shift_range-kernel_W//2
        W_end = W_s//2+self.shift_range+kernel_W//2
        H_start = H_s//2-self.shift_range-kernel_H//2
        H_end = H_s//2+self.shift_range+kernel_H//2
        assert W_start >= 0 and W_end <= W_s, 'input of conv crop w error!!!'
        assert H_start >= 0 and H_end <= H_s, 'input of conv crop h error!!!'
        sat_feature = sat_feature[:,:,H_start:H_end,W_start:W_end]
        _,_,H_i,W_i = sat_feature.size()
        if test_method != 2: 
            sat_feature = F.normalize(sat_feature.reshape(M,-1)) 
            sat_feature = sat_feature.view(M,C,H_i,W_i)
        
        # corrilation to get similarity matrix
        if 0:
            correlation_matrix = torch.tensor([], device=grd_feature.device)
            for i in range(N):
                kernel = grd_feature[i].unsqueeze(0)#[C,H,W]->[1,C,H,W] # out_ch, in_ch
                correlation = F.conv2d(sat_feature, kernel) # M, 1, 2*shift_rang+1, 2*shift_rang+1)
                correlation_matrix = torch.cat([correlation_matrix, correlation], dim=1) ##[M.N,2*shift_rang+1, 2*shift_rang+1]
        else:
            # speed up, but memory not enough
            in_feature = sat_feature.repeat(1,N,1,1) #[M,C,H,W]->[M,N*C,H,W]
            correlation_matrix = F.conv2d(in_feature, grd_feature,groups=N) # M, N, 2*shift_rang+1, 2*shift_rang+1)
        
        if test_method != 2:
            partical = F.avg_pool2d(sat_feature.pow(2), (kernel_H,kernel_W), stride=1, divisor_override=1) #[M,C,2*shift_rang+1, 2*shift_rang+1]
            partical = torch.sum(partical, dim=1).unsqueeze(1) # sum on C
            partical = torch.maximum(partical, torch.ones_like(partical) * 1e-7) # for /0
            assert torch.all(partical!=0.), 'have 0 in partical!!!'
            if test_method == 0:
                correlation_matrix /= torch.sqrt(partical)
            else:
                correlation_matrix /= partical 
            similarity_matrix = torch.amax(correlation_matrix,dim=(2,3)) #M,N
            
            #if torch.max(similarity_matrix) > 1:
            #    print('>1,parical:',partical.cpu().detach(), correlation_matrix.cpu().detach())
        else:
            W = correlation_matrix.size()[-1]
            max_index = torch.argmax(correlation_matrix.view(M,N,-1),dim=-1) #M,N
            max_pos = torch.cat([(max_index//W).unsqueeze(-1), (max_index%W).unsqueeze(-1)], dim=-1)#M,N,2
            
            # crop sat, and normalize
            in_feature = torch.tensor([],device=sat_feature.device)
            for i in range(N):
                sat_f_n = torch.tensor([],device=sat_feature.device)
                for j in range(M):
                    sat_f = sat_feature[j,:,max_pos[j,i,0]:max_pos[j,i,0]+H_k,max_pos[j,i,1]:max_pos[j,i,1]+W_k] # [C,H,W]
                    sat_f_n = torch.cat([sat_f_n,sat_f.unsqueeze(0)], dim=0) # [M,C,H,W]
                in_feature = torch.cat([in_feature,sat_f_n.unsqueeze(1)], dim=1) # [M,N,C,H,W]
            
            in_feature = F.normalize(in_feature.reshape(M*N,-1)) 
            in_feature = in_feature.view(M,N*C,H_k,W_k)
            
            grd_feature = F.normalize(grd_feature.reshape(N,-1)) 
            grd_feature = grd_feature.view(N,C,H_k,W_k)
            
            similarity_matrix = F.conv2d(in_feature, grd_feature,groups=N) # M, N, 1,1)
            similarity_matrix = similarity_matrix.view(M,N)
            
        #print('similarity_matric max&min:',torch.max(similarity_matrix).item(), torch.min(similarity_matrix).item() )
        return similarity_matrix, sat_f
    
# partial correlation similarity loss function, the input 
# inputs:
#   grd_feature, sat_feature, margin
# return:
#   loss, distance_positive.mean(), distance_negative.mean()
class partical_similarity_loss_bak(nn.Module):
    def __init__(self, shift_range):
        super(partical_similarity_loss_bak, self).__init__()
        self.similarity_function = partical_similarity(shift_range)
    
    def forward(self, sat_feature, grd_feature, margin=1, angle_label=None, angle_pred=None, sat_global_inv=None):
        B = grd_feature.size()[0]
        
        distance_negative =  2-2*self.similarity_function(grd_feature, sat_feature) # range: 2~0
        #distance_negative *=25 #original use of alpha=5 range: 100~0
        
        ### distance rectification factor - beta
        margin = torch.tensor(margin, device=grd_feature.device)
        #margin *= 25 #original use of alpha=5
        beta = margin/2.0
        
        distance_positive = torch.diagonal(distance_negative)

        
        ### rectified distance for computing weight mask
        dist_rec = distance_positive.repeat(B,1).t() - distance_negative + beta
    
        p = 1.0/(1.0 + torch.exp( dist_rec ))
        
        ### weight mask generating 
        w_mask = F.relu(-torch.log2(p + 0.00000001))
        
        ### weight mask pruningatten = self.attention(q, k, value=src, attn_mask=mask, key_padding_mask=srcKeyPaddingMask)
        w_low = -torch.log2(1.0/(1.0 + torch.exp(  -1.0*margin + beta ) + 0.00000001) )
        w_high = -torch.log2(1.0/(1.0 + torch.exp(  -0.0*margin + beta ) + 0.00000001) )
        
        w_mask[w_mask<w_low] = 0.1/B # pruning over simple data
        
        w_mask[w_mask>w_high] = w_high # pruning over extreme hard data
        
        
        # diagonal elements need to be neglected (set to zero)
        w_mask = w_mask - torch.diag(torch.diagonal(w_mask))
        
        # main loss computing
        losses = w_mask * torch.log(1.0 + torch.exp( (distance_positive.repeat(B,1).t() - distance_negative)))
        loss = losses.mean()
                
        ### orientation regression loss 
        '''
        if angle_pred != None:
            if 1:
                losses_OR = (angle_pred-torch.cos(angle_label)).pow(2)
                print("losses_OR: ", losses_OR.detach().cpu().numpy())
                losses_OR = losses_OR.repeat(grd_global.size()[0],1).t()
    
                # OR loss computing
                losses_OR = w_mask * losses_OR 
                
                #loss combining, as a recommendation - theta1 : theta2 = 2 : 1 (here theta2 can be a number in {1,2,3,...,10})
                losses = theta1*losses + theta2*losses_OR 
                loss = losses.mean()
            else:
                losses_OR = (angle_pred-torch.cos(angle_label)).pow(2)
                print("losses_OR: ", losses_OR.detach().cpu().numpy())
                loss = theta1*losses.mean()+theta2*losses_OR.mean()
        else:
            loss = losses.mean()
        '''
            
        
        ###### exp based los
        return loss, distance_positive.mean(), distance_negative.mean()
    
class partical_similarity_loss(nn.Module):
    """
    CVM
    """
    ### the value of margin is given according to the facenet
    def __init__(self, shift_range, loss_weight=10.0):
        super(partical_similarity_loss, self).__init__()
        self.similarity_function = partical_similarity(shift_range)
        self.loss_weight = loss_weight
        
    def forward(self, sat_feature, grd_feature, margin=1, angle_label=None, angle_pred=None, sat_global_inv=None):
        B = grd_feature.size()[0]

        if 1:
            dist_array =  2-2*self.similarity_function(grd_feature, sat_feature) # range: 2~0
        else:
            sat_feature = sat_feature.view(B,-1)
            grd_feature = grd_feature.view(B,-1)
            sat_feature = F.normalize(sat_feature) 
            grd_feature = F.normalize(grd_feature) 
            dist_array =  2 - 2*sat_feature@grd_feature.T

        pos_dist = torch.diagonal(dist_array)
        
        # ground to satellite
        triplet_dist_g2s = pos_dist - dist_array
        triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
        triplet_dist_g2s = triplet_dist_g2s - torch.diag(torch.diagonal(triplet_dist_g2s))
        top_k_g2s, _ = torch.topk((triplet_dist_g2s.t()), B)
        loss_g2s = torch.mean(top_k_g2s)
        
        # satellite to ground
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
        triplet_dist_s2g = triplet_dist_s2g - torch.diag(torch.diagonal(triplet_dist_s2g))
        top_k_s2g, _ = torch.topk(triplet_dist_s2g, B)
        loss_s2g = torch.mean(top_k_s2g)
        
        loss = (loss_g2s + loss_s2g) / 2.0
        #loss = loss_g2s
            
        pos_dist_avg = pos_dist.mean()
        nega_dist_avg = dist_array.mean()

        return loss, pos_dist_avg, nega_dist_avg.sum()   