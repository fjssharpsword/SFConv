# encoding: utf-8
"""
Training implementation for LIDC-IDRI CT dataset - Segmentation - 3D UNet
Author: Jason.Fang
Update time: 17/07/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from thop import profile
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
#define by myself
from utils.common import dice_coeff, count_bytes, transparent_back
from dsts.fundus_seg import get_train_dataloader, get_test_dataloader
from nets.unet_2d import UNet, DiceLoss
from nets.pkgs.factorized_conv import weightdecay

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
BATCH_SIZE = 16
MAX_EPOCHS = 200
UNET_PARAMS = ['module.up1.conv.double_conv.0.weight', #torch.Size([512, 1024, 3, 3])
               'module.up1.conv.double_conv.0.P', #torch.Size([1536, 512])
               'module.up1.conv.double_conv.0.Q'] #torch.Size([512, 3072])
CKPT_PATH = '/data/pycode/SFConv/ckpts/fundus_seg_unet_sfconv.pkl'
def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    print ('==>>> total trainning batch number: {}'.format(len(dataloader_train)))
    print ('==>>> total test batch number: {}'.format(len(dataloader_test)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = UNet(n_channels=3, n_classes=1).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Vin-CXR dataset: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    criterion = DiceLoss().cuda() #nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')
    #save the initialing data of weights
    #for name, param in model.named_parameters():
    #    if name in UNET_PARAMS:
    #        print(name,'---', param.size())

    print('********************begin training!********************')
    log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, mask) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_out = model(var_image)
                loss_tensor = criterion(var_out, var_mask)

                optimizer_model.zero_grad()
                loss_tensor.backward()
                weightdecay(model, coef=1E-4) #weightdecay for factorized_conv
                optimizer_model.step()#update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        #test
        model.eval()
        test_loss = []
        with torch.autograd.no_grad():
            for batch_idx,  (image, mask) in enumerate(dataloader_test):
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_out = model(var_image)
                loss_tensor = criterion(var_out, var_mask)
                test_loss.append(loss_tensor.item())
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        print("\r Eopch: %5d test loss = %.6f" % (epoch + 1, np.mean(test_loss)) )

        #save checkpoint with lowest loss 
        if loss_min > np.mean(test_loss):
            loss_min = np.mean(test_loss)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            #torch.save(model.state_dict(), CKPT_PATH)
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

        #print the loss
        log_writer.add_scalars('DiceLoss/Fundus_Seg_UNet_Conv', {'train':np.mean(train_loss), 'val':np.mean(test_loss)}, epoch+1)
    log_writer.close() #shut up the tensorboard
    print("\r Dice of testset = %.4f" % (1-loss_min))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=8, shuffle=False, num_workers=1) #BATCH_SIZE
    print ('==>>> total test batch number: {}'.format(len(dataloader_test)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = UNet(n_channels=3, n_classes=1).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Vin-CXR dataset: "+CKPT_PATH)
    model.eval()
    criterion = DiceLoss().cuda()
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    time_res = []
    dice_coe = []
    with torch.autograd.no_grad():
        for batch_idx, (image, mask) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)
            pred = var_out.cpu().data
            #pred = torch.where(var_out.cpu().data>0.5, 1, 0)
            dice_coe.append(criterion(pred, mask).item())
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #model
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    #Compute Dice coefficient
    print("\r Dice coefficient = %.4f" % (1-np.mean(dice_coe)))

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()