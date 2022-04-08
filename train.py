#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

@author: Shan
"""
import random

import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, Recall, F1Score, Precision

from utils.dataloader import load_train_data, load_val_data, bev_to_cam
from model import Net
import datetime
from torchvision import transforms

class focal_loss(nn.Module):
    def __init__(self, gamma=2., alpha=1.): # gamma=2., alpha=.25
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        # pt_1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        # pt_0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        # return -torch.mean(self.alpha * torch.pow(1. - pt_1, self.gamma) * torch.log(pt_1)) - torch.mean(
        #     (1 - self.alpha) * torch.pow(pt_0, self.gamma) * torch.log(1. - pt_0))

        # changed by shan
        pt = torch.where(y_true == 1, y_pred, 1-y_pred)
        return -torch.mean(self.alpha*torch.pow(1. - pt, self.gamma) * torch.log(pt+1e-8))

###### learning criterion assignment #######
def Train( net, args, tb_writer):

    criterion =  focal_loss()
    criterion.cuda()

    bestResult = 0.

    #optimizer = optim.SGD(net.parameters(), lr=args.lr)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)#, last_epoch=8000)

    # loop over the dataset multiple times
    for epoch in range(0, args.epochs):
        net.train()
        torch.cuda.empty_cache()

        trainloader = load_train_data(args.batch, args.sequence)
        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs
            mask, left, right = Data

            # not have enough sequence
            if left.size(1) < args.sequence:
                continue

            mask = mask.cuda()
            left = left.cuda()
            right = right.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            pre_mask = net.forward(left, right)
            if 0: #debug
                img = transforms.functional.to_pil_image(pre_mask[0], mode='L')
                img.save('pre_mask.png')
                img = transforms.functional.to_pil_image(mask[0], mode='L')
                img.save('mask.png')

            # turn pre_mask to ori size
            pre_mask = bev_to_cam(pre_mask, mask.shape[-2:])

            if 0: #debug
                img = transforms.functional.to_pil_image(pre_mask[0], mode='L')
                img.save('pre_tran_mask.png')

            loss = criterion(mask,pre_mask)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # Write losses to tensorboard
            # Update avg meters
            tb_writer.add_scalar("loss/train", loss.item(), Loop)

            if Loop % 5 == 4:   # print every 64*5 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch, Loop, loss.item()))

        scheduler.step()
                
        ### save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        if not os.path.exists(args.experiment):
            os.makedirs(args.experiment)
        torch.save(net.state_dict(), args.experiment + 'model_' + str(compNum+1) + '.pth')
     
        ## test
        Val(net, args, tb_writer, bestResult, epoch)

    
    print('Finished Training')


def Val(net, args, tb_writer, best_result, epoch):
    ### net evaluation state
    net.eval()
    torch.cuda.empty_cache()

    metric_acc = Accuracy()
    metric_precision = Precision()
    metric_recall = Recall()
    metric_f1score = F1Score()


    testloader = load_val_data(args.batch, args.sequence)

    gt = torch.tensor([])
    pre = torch.tensor([])
    for i, data in enumerate(testloader, 0):
        mask, left, right = data

        left = left.cuda()
        right = right.cuda()

        pre_mask = net.forward(left, right)

        if 0:  # debug
            img = transforms.functional.to_pil_image(pre_mask[0], mode='L')
            img.save('pre_mask.png')

        # turn pre_mask to ori size
        pre_mask = bev_to_cam(pre_mask, mask.shape[-2:])

        if 0:  # debug
            img = transforms.functional.to_pil_image(pre_mask[0], mode='L')
            img.save('pre_tran_mask.png')
            img = transforms.functional.to_pil_image(mask[0], mode='L')
            img.save('mask.png')

        # mask = mask.flatten().int()
        # pre_mask = (pre_mask>=0.5).int().flatten().detach().cpu()
        mask = mask.flatten().int()
        pre_mask = pre_mask.flatten().detach().cpu()

        metric_acc.update(pre_mask, mask)
        metric_precision.update(pre_mask, mask)
        metric_recall.update(pre_mask, mask)
        metric_f1score.update(pre_mask, mask)

        # gt = torch.cat((gt, mask))
        # pre = torch.cat((pre, pre_mask))

    acc = metric_acc.compute()
    print(f"Accuracy on all data: {acc}")
    precision = metric_precision.compute()
    print(f"Precision on all data: {precision}")
    recall = metric_recall.compute()
    print(f"Recall on all data: {recall}")
    f1score = metric_f1score.compute()
    print(f"F1score on all data: {f1score}")

    # Write val losses to tensorboard
    tb_writer.add_scalar("acc/val", acc.item(), epoch)
    tb_writer.add_scalar("precision/val", precision.item(), epoch)
    tb_writer.add_scalar("recall/val", recall.item(), epoch)
    tb_writer.add_scalar("f1score/val", f1score.item(), epoch)

    # target_names = ['water', 'other']
    # print(classification_report(pre, gt, target_names=target_names))

    net.train()

    ### restore the best params
    if (f1score.item() > best_result):
        if not os.path.exists(args.experiment):
            os.makedirs(args.experiment)
        torch.save(net.state_dict(), args.experiment + 'Model_best.pth')

    return f1score.item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=1, help='resume the trained model.1: whole model,2:only feature get part')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') #1e-2
    parser.add_argument('--batch', type=int, default=16, help='batch size')  # 1e-2
    parser.add_argument('--sequence', type=int, default=16, help='sequence of rgb images')
    
    parser.add_argument('--experiment', type=str, default='weights/', help='dir to save net weights')

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    # test to load 1 data
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.deterministic = True
    args = parse_args()

    # tensorboard summary_writer
    tb_writer = SummaryWriter(log_dir=args.experiment)

    ####################### model assignment #######################
    net = Net() # add model !!!!!!!!!!!!!!!

    if args.resume == 1:
        net.load_state_dict(torch.load(args.experiment+'Model_best.pth'))
        print("resume finished")

    net.cuda()
    ###########################

    if args.test:
        Val(net, args, tb_writer, 0., 0)
    else:
        Train(net, args, tb_writer)

    
