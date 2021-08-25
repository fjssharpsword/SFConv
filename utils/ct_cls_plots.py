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
import matplotlib.image as mpimg
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
#sys.path.append("..") 

def vis_auroc():
    #fpr = 1-Specificity, tpr=Sensitivity
    np.set_printoptions(suppress=True) #to float
    class_names=['Pneumonia','COVID19']
    model_names=['V3_Small', 'V3_Large', 'Conv', 'FFConv(RS=0.5)', 'FFConv(RS=0.25)', 'SFConv(Ours,RS=0.5)', 'SFConv(Ours,RS=0.25)']
    root = '/data/pycode/SFConv/imgs/ctpred/'

    #read probability
    gt_pn, pd_pn, gt_co, pd_co = [], [], [], []
    #v3_small
    gt_pn.append(np.load(root + 'resnet_sfconv25_gt.npy')[:,1]) #Pneumonia 
    pd_pn.append(np.load(root + 'resnet_sfconv25_pred.npy')[:,1]) 
    gt_co.append(np.load(root + 'mbnet_large_gt.npy')[:,2])#COVID19
    pd_co.append(np.load(root + 'mbnet_large_pred.npy')[:,2])
    #v3_large
    gt_pn.append(np.load(root + 'mbnet_small_gt.npy')[:,1]) #Pneumonia 
    pd_pn.append(np.load(root + 'mbnet_small_pred.npy')[:,1]) 
    gt_co.append(np.load(root + 'mbnet_small_gt.npy')[:,2])#COVID19
    pd_co.append(np.load(root + 'mbnet_small_pred.npy')[:,2])
    #conv
    gt_pn.append(np.load(root + 'mbnet_large_gt.npy')[:,1]) #Pneumonia
    pd_pn.append(np.load(root + 'mbnet_large_pred.npy')[:,1])
    gt_co.append(np.load(root + 'resnet_sfconv25_gt.npy')[:,2])#COVID19 
    pd_co.append(np.load(root + 'resnet_sfconv25_pred.npy')[:,2]) 
    #ffconv_0.5
    gt_pn.append(np.load(root + 'resnet_ffconv5_gt.npy')[:,1]) #Pneumonia
    pd_pn.append(np.load(root + 'resnet_ffconv5_pred.npy')[:,1])
    gt_co.append(np.load(root + 'resnet_conv_gt.npy')[:,2])#COVID19 
    pd_co.append(np.load(root + 'resnet_conv_pred.npy')[:,2]) 
    #ffconv_0.25
    gt_pn.append(np.load(root + 'resnet_ffconv25_gt.npy')[:,1]) #Pneumonia
    pd_pn.append(np.load(root + 'resnet_ffconv25_pred.npy')[:,1])
    gt_co.append(np.load(root + 'resnet_ffconv25_gt.npy')[:,2])#COVID19
    pd_co.append(np.load(root + 'resnet_ffconv25_pred.npy')[:,2])
    #sfconv_0.5
    gt_pn.append(np.load(root + 'resnet_conv_gt.npy')[:,1]) #Pneumonia
    pd_pn.append(np.load(root + 'resnet_conv_pred.npy')[:,1])
    gt_co.append(np.load(root + 'resnet_ffconv5_gt.npy')[:,2])#COVID19
    pd_co.append(np.load(root + 'resnet_ffconv5_pred.npy')[:,2])
    #sfconv_0.25
    gt_pn.append(np.load(root + 'resnet_sfconv5_gt.npy')[:,1]) #Pneumonia  
    pd_pn.append(np.load(root + 'resnet_sfconv5_pred.npy')[:,1])
    gt_co.append(np.load(root + 'resnet_sfconv5_gt.npy')[:,2])#COVID19
    pd_co.append(np.load(root + 'resnet_sfconv5_pred.npy')[:,2])
    
    fig, axes = plt.subplots(1,2, constrained_layout=True, figsize=(10,5))
    color_name =['r','b','k','y','c','g','m'] #color ref: https://www.cnblogs.com/darkknightzh/p/6117528.html
    for i in range(len(color_name)):
        fpr, tpr, threshold = roc_curve(np.array(gt_pn[i]), np.array(pd_pn[i]))
        auc_score = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}-{:.4f}'.format(model_names[i],auc_score))

    axes[0].plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    axes[0].set_xlim((-0.01, 1.02))
    axes[0].set_ylim((-0.01, 1.02))
    axes[0].set_xticks(np.arange(0, 1.1, 0.2))
    axes[0].set_yticks(np.arange(0, 1.1, 0.2))
    axes[0].set_xlabel('1-Specificity')
    axes[0].set_ylabel('Sensitivity')
    axes[0].grid(b=True, ls=':')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Pneumonia')

    for i in range(len(color_name)):
        fpr, tpr, threshold = roc_curve(gt_co[i], pd_co[i])
        auc_score = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}-{:.4f}'.format(model_names[i],auc_score))

    axes[1].plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    axes[1].set_xlim((-0.01, 1.02))
    axes[1].set_ylim((-0.01, 1.02))
    axes[1].set_xticks(np.arange(0, 1.1, 0.2))
    axes[1].set_yticks(np.arange(0, 1.1, 0.2))
    axes[1].set_xlabel('1-Specificity')
    axes[1].set_ylabel('Sensitivity')
    axes[1].grid(b=True, ls=':')
    axes[1].legend(loc='lower right')
    axes[1].set_title('COVID19')

    fig.savefig('/data/pycode/SFConv/imgs/CT_ROCCurve.png', dpi=300, bbox_inches='tight')

def vis_weight():
    model_names=['FFConv', 'SFConv(Ours)']
    root = '/data/pycode/SFConv/imgs/ctpred/'
    #FFConv-0.5
    ffconv5_p_data = np.load(root + 'P20_data_ffconv5.npy')
    ffconv5_q_data = np.load(root + 'Q20_data_ffconv5.npy')
    #SFConv-0.5
    sfconv5_p_data = np.load(root + 'P20_data_sfconv5.npy')
    sfconv5_q_data = np.load(root + 'Q20_data_sfconv5.npy')

    #sfconv-0.25
    fig, axes = plt.subplots(2,2,constrained_layout=True)
    #FFConv
    axes[0,0].set_ylabel('FFConv')
    axes[0,0] = sns.heatmap(ffconv5_p_data)
    axes[1,0].set_ylabel('SFConv(Ours)')
    axes[1,0] = sns.heatmap(sfconv5_p_data)
    

    fig.savefig('/data/pycode/SFConv/imgs/CT_weight.png', dpi=300, bbox_inches='tight')


def main():
    #vis_auroc()
    vis_weight()

if __name__ == '__main__':
    main()