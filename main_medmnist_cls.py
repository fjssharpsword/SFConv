# encoding: utf-8
"""
Training implementation for Medical-MNIST dataset.  
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
from utils.common import count_bytes
from nets.resnet import resnet18
from nets.densenet import densenet121
from nets.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from nets.pkgs.factorized_conv import weightdecay
from dsts.med_mnist_cls import get_dataloader_train, get_dataloader_test

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 50
batch_size = 256
RESNET_PARAMS = ['module.4.0.conv1.weight', 'module.4.0.conv1.P', 'module.4.0.conv1.Q',\
                 'module.5.0.conv1.weight', 'module.5.0.conv1.P', 'module.5.0.conv1.Q', \
                 'module.6.0.conv1.weight', 'module.6.0.conv1.P', 'module.6.0.conv1.Q', \
                 'module.7.0.conv1.weight', 'module.7.0.conv1.P', 'module.7.0.conv1.Q' ]

CKPT_PATH = '/data/pycode/SFConv/ckpts/medmnist_mbnet_large.pkl'
def Train():
    print('********************load data********************')
    train_loader = get_dataloader_train(batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = get_dataloader_test(batch_size=batch_size, shuffle=False, num_workers=1)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = mobilenet_v3_large(pretrained=False, num_classes=6).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
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

        #test
        model.eval()
        loss_test = []
        total_cnt, correct_cnt = 0, 0
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(test_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                loss_tensor = criterion.forward(var_out, var_label)
                loss_test.append(loss_tensor.item())
                _, pred_label = torch.max(var_out.data, 1)
                total_cnt += var_image.data.size()[0]
                correct_cnt += (pred_label == var_label.data).sum()
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        acc = correct_cnt * 1.0 / total_cnt
        print("\r Eopch: %5d val loss = %.6f, ACC = %.6f" % (epoch + 1, np.mean(loss_test), acc) )

        # save checkpoint
        if acc_min < acc:
            acc_min = acc
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        #log_writer.add_scalars('CrossEntropyLoss/Medical-MNIST-ResNet', {'Conv':np.mean(loss_train)}, epoch+1)
    #log_writer.close() #shut up the tensorboard

def Test():
    print('********************load data********************')
    test_loader = get_dataloader_test(batch_size=batch_size, shuffle=False, num_workers=1)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = mobilenet_v3_large(pretrained=False, num_classes=6).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')

    print('********************begin Testing!********************')
    total_cnt, correct_cnt = 0, 0 
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(test_loader):
            #forward
            var_image = torch.autograd.Variable(img).cuda()
            var_label = torch.autograd.Variable(lbl).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)
            _, pred_label = torch.max(var_out.data, 1)
            total_cnt += var_image.data.size()[0]
            correct_cnt += (pred_label == var_label.data).sum()
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    acc = correct_cnt * 1.0 / total_cnt
    ci  = 1.96 * math.sqrt( (acc * (1 - acc)) / total_cnt) #1.96-95%
    print("\r test ACC/CI = %.4f/%.4f" % (acc, ci) )
    
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()
