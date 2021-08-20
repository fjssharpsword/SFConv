# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 29/07/2021
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
import math
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
import torchvision.datasets as dset
#sys.path.append("..") 
from vincxr_det import get_box_dataloader_VIN

def calc(data):
    n=len(data)
    niu=0.0 # niu表示平均值,即期望.
    niu2=0.0 # niu2表示平方的平均值
    niu3=0.0 # niu3表示三次方的平均值
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n  
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu,sigma,niu3]

def calc_stat(data):
    [niu, sigma, niu3]=calc(data)
    n=len(data)
    niu4=0.0 # niu4计算峰度计算公式的分子
    for a in data:
        a -= niu
        niu4 += a**4
    niu4 /= n

    skew =(niu3 -3*niu*sigma**2-niu**3)/(sigma**3) # 偏度计算公式
    kurt=niu4/(sigma**4) # 峰度计算公式:下方为方差的平方即为标准差的四次方
    return [niu, sigma,skew,kurt]

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
 
    print('********************load data********************')
    cxr_set = get_box_dataloader_VIN(batch_size=2, shuffle=True, num_workers=0)
    trans = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    voc_set = dset.VOCSegmentation(root='/data/fjsdata/VOC2012/', year='2012', image_set='val', transform=trans, download=False) 
    voc_set = torch.utils.data.DataLoader(dataset=voc_set,batch_size= 2,shuffle= True, num_workers=0 , collate_fn=collate_fn)
    CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    print('********************load data succeed!********************')

    print('*******Plot!*********')
    fig, axes = plt.subplots(3,2, constrained_layout=True) #figsize=(6,9)

    for batch_idx, (images, masks) in enumerate(voc_set):
        voc_img0 = images[0].numpy().flatten()
        voc_img1 = images[1].numpy().flatten()
        [_, _, voc_skew0, voc_kurt0] = calc_stat(voc_img0)
        [_, _, voc_skew1, voc_kurt1] = calc_stat(voc_img1)
        voc_info0 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(voc_skew0, voc_kurt0) # 标注
        voc_info1 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(voc_skew1, voc_kurt1)
        sns.distplot(voc_img0, kde=True, ax=axes[0,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[0,0].set_xlabel(voc_info0)
        axes[0,0].set_ylabel('Natural Images')
        sns.distplot(voc_img1, kde=True, ax=axes[0,1], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[0,1].set_xlabel(voc_info1)
        break

    for batch_idx, (images, targets) in enumerate(cxr_set):
        cxr_img0 = images[0].numpy().flatten()
        cxr_img1 = images[1].numpy().flatten()
        [_, _, cxr_skew0, cxr_kurt0] = calc_stat(cxr_img0)
        [_, _, cxr_skew1, cxr_skew1] = calc_stat(cxr_img1)
        cxr_info0 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(cxr_skew0, cxr_kurt0) # 标注
        cxr_info1 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(cxr_skew1, cxr_skew1)
        sns.distplot(cxr_img0, kde=True, ax=axes[1,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[1,0].set_xlabel(cxr_info0)
        axes[1,0].set_ylabel('Medical Images')
        sns.distplot(cxr_img1, kde=True, ax=axes[1,1], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[1,1].set_xlabel(cxr_info1)


        cxr_box0 = targets[0]['boxes'].numpy()[0]
        cxr_box0 = images[0].numpy()[:,int(cxr_box0[0]):int(cxr_box0[2]),int(cxr_box0[1]):int(cxr_box0[3])]
        cxr_box0 = cxr_box0.flatten()
        cxr_box1 = targets[1]['boxes'].numpy()[0]
        cxr_box1 = images[1].numpy()[:,int(cxr_box1[0]):int(cxr_box1[2]),int(cxr_box1[1]):int(cxr_box1[3])]
        cxr_box1 = cxr_box1.flatten()
        [_, _, box_skew0, box_kurt0] = calc_stat(cxr_box0)
        [_, _, box_skew1, box_kurt1] = calc_stat(cxr_box1)
        box_info0 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(box_skew0, box_kurt0) # 标注
        box_info1 = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(box_skew1, box_kurt1)
        sns.distplot(cxr_box0, kde=True, ax=axes[2,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[2,0].set_xlabel(box_info0)
        axes[2,0].set_ylabel('Lesion regions')
        sns.distplot(cxr_box1, kde=True, ax=axes[2,1], hist_kws={'color':'green'}, kde_kws={'color':'red'})
        axes[2,1].set_xlabel(box_info1)

        break

    fig.savefig('/data/pycode/SFConv/imgs/data_dis.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()