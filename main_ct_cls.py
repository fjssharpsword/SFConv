# encoding: utf-8
"""
Training implementation for COVIDx-CT dataset.  
Author: Jason.Fang
Update time: 2w/08/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from thop import profile
from tensorboardX import SummaryWriter
#define by myself
from utils.common import count_bytes, compute_AUCs
from nets.resnet import resnet18
from nets.densenet import densenet121
from nets.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from nets.pkgs.factorized_conv import weightdecay
from dsts.COVIDx_ct import get_dataloader_train, get_dataloader_val, get_dataloader_test

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 20 #50
batch_size = 256
CLASS_NAMES = ['Normal','Pneumonia','COVID19']
RESNET_PARAMS = ['layer1.0.conv1.P', 'layer1.0.conv1.Q']
               #['module.layer4.1.conv2.weight', 'module.layer4.1.conv2.P', 'module.layer4.1.conv2.Q']
DATA_PATH = '/data/pycode/SFConv/imgs/ctpred/'
CKPT_PATH = '/data/pycode/SFConv/ckpts/ct_resnet_ffconv5.pkl'

def Train():
    print('********************load data********************')
    train_loader = get_dataloader_train(batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = get_dataloader_val(batch_size=batch_size, shuffle=False, num_workers=8)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total test batch number: {}'.format(len(val_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=len(CLASS_NAMES)).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.BCELoss().cuda() #nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    acc_min = 0.50 #float('inf')
    for epoch in range(max_epoches):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , max_epoches))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(train_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                # backward and update parameters
                optimizer_model.zero_grad()
                loss_tensor = criterion.forward(var_out, var_label) 
                loss_tensor.backward()
                weightdecay(model, coef=1E-4) #weightdecay for factorized_conv
                optimizer_model.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))
        """
        #print weight data and grad
        if epoch + 1 == 1 or epoch + 1 == max_epoches:
            for name, param in model.named_parameters():
                if param.requires_grad and name in RESNET_PARAMS:
                    np.save(DATA_PATH + name.split('.')[-1] + str(epoch+1) + '_grad_sfconv25.npy', param.grad.cpu().data.numpy())
                    np.save(DATA_PATH + name.split('.')[-1] + str(epoch+1) + '_data_sfconv25.npy', param.clone().cpu().data.numpy())
        """
        #test
        model.eval()
        loss_test = []
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(val_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                loss_tensor = criterion.forward(var_out, var_label)
                loss_test.append(loss_tensor.item())
                gt = torch.cat((gt, lbl), 0)
                pred = torch.cat((pred, var_out.data.cpu()), 0)
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        acc = np.mean(compute_AUCs(gt, pred, len(CLASS_NAMES)))
        print("\r Eopch: %5d val loss = %.6f, AUROC = %.6f" % (epoch + 1, np.mean(loss_test), acc) )

        # save checkpoint
        if acc_min < acc:
            acc_min = acc
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    test_loader = get_dataloader_test(batch_size=batch_size, shuffle=False, num_workers=8)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=len(CLASS_NAMES)).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')

    for name, param in model.named_parameters():
        if name in RESNET_PARAMS:
            np.save(DATA_PATH + name.split('.')[-1] + '_ffconv.npy', param.clone().cpu().data.numpy())

    print('********************begin Testing!********************')
    time_res = []
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(test_loader):
            #forward
            var_image = torch.autograd.Variable(img).cuda()
            var_label = torch.autograd.Variable(lbl).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)

            gt = torch.cat((gt, lbl), 0)
            pred = torch.cat((pred, var_out.data.cpu()), 0)

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    AUROCs = compute_AUCs(gt, pred, len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
    print('The average AUROC is {:.4f}'.format(np.mean(AUROCs)))
    #save
    np.save(DATA_PATH + 'resnet_sfconv25_gt.npy',gt.numpy()) #np.load()
    np.save(DATA_PATH + 'resnet_sfconv25_pred.npy',pred.numpy())

def main():
    #Train()
    Test()

if __name__ == '__main__':
    main()
