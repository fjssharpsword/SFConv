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
    cxr_set = get_box_dataloader_VIN(batch_size=1, shuffle=True, num_workers=0)
    trans = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    voc_set = dset.VOCSegmentation(root='/data/fjsdata/VOC2012/', year='2012', image_set='val', transform=trans, download=False) 
    voc_set = torch.utils.data.DataLoader(dataset=voc_set,batch_size= 1,shuffle= True, num_workers=0 , collate_fn=collate_fn)
    CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    print('********************load data succeed!********************')

    print('*******Plot!*********')
    for batch_idx, (images, masks) in enumerate(voc_set):
        voc_img = images[0].numpy().flatten()
        break
    for batch_idx, (images, targets) in enumerate(cxr_set):
        cxr_img = images[0].numpy().flatten()
        break

    [_, _, voc_skew, voc_kurt] = calc_stat(voc_img)
    [_, _, cxr_skew, cxr_kurt] = calc_stat(cxr_img)
    voc_info = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(voc_skew, voc_kurt) # 标注
    cxr_info = r'$\ Skewness=%.2f,\ kurtosis=%.2f$' %(cxr_skew, cxr_kurt)

    plt.text(0,1,voc_info,bbox=dict(facecolor='red',alpha=0.25))
    plt.text(0,2,cxr_info,bbox=dict(facecolor='blue',alpha=0.25))
    voc_his = plt.hist(voc_img, normed=True, facecolor='r',alpha=0.9)
    cxr_his = plt.hist(cxr_img, normed=True, facecolor='b',alpha=0.7)
    plt.grid(True)
    plt.title('Distributions of visual features')
    plt.savefig('/data/pycode/SFConv/imgs/data_dis.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()