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
    ffconv_p_data = np.load(root + 'P_ffconv.npy')
    ffconv_q_data = np.load(root + 'Q_ffconv.npy')
    ffconv_w_data = np.dot(ffconv_p_data, ffconv_q_data)
    #SFConv-0.5
    sfconv_p_data = np.load(root + 'P_sfconv.npy')
    sfconv_q_data = np.load(root + 'Q_sfconv.npy')
    sfconv_w_data = np.dot(sfconv_p_data, sfconv_q_data)

    ffconv_p_data = ffconv_p_data.flatten()
    ffconv_q_data = ffconv_q_data.flatten()
    sfconv_p_data = sfconv_p_data.flatten()
    sfconv_q_data = sfconv_q_data.flatten()

    #calculate svd
    _, s_ffconv, _ = np.linalg.svd(ffconv_w_data, full_matrices=True)
    _, s_sfconv, _ = np.linalg.svd(sfconv_w_data, full_matrices=True)
    #explained variance
    var_ff = np.round(s_ffconv**2/np.sum(s_ffconv**2), decimals=3)
    var_ff = var_ff[np.nonzero(var_ff)]
    var_sf = np.round(s_sfconv**2/np.sum(s_sfconv**2), decimals=3)
    var_sf = var_sf[np.nonzero(var_sf)]

    #sfconv-0.25
    fig, axes = plt.subplots(2,3,constrained_layout=True)
    #FFConv
    axes[0,0].set_ylabel('SFConv (Ours)')
    axes[0,0].set_title('P, shape=[192,32]')
    sns.distplot(ffconv_p_data, kde=True, ax=axes[0,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    info = r'$\ variance=%.2f$' %(np.var(ffconv_p_data))
    axes[0,0].set_xlabel(info)

    axes[0,1].set_title('Q, shape=[32,192]')
    sns.distplot(ffconv_q_data, kde=True, ax=axes[0,1], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    info = r'$\ variance=%.2f$' %(np.var(ffconv_q_data))
    axes[0,1].set_xlabel(info)

    axes[0,2].set_title('W, shape=[192,192]')
    sns.barplot(x=list(range(1,len(var_sf)+1)), y=var_sf, color="limegreen", ax =axes[0,2] )
    axes[0,2].set_ylabel('Explained variance (%)')
    axes[0,2].set_xlabel('Number of non-zero SVs')
    for ind, label in enumerate(axes[0,2].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif (ind+1) % 8 == 0:  # every 4th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    #SFConv
    axes[1,0].set_ylabel('FFConv')
    sns.distplot(sfconv_p_data, kde=True, ax=axes[1,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    info = r'$\ variance=%.2f$' %(np.var(sfconv_p_data))
    axes[1,0].set_xlabel(info)

    sns.distplot(sfconv_q_data, kde=True, ax=axes[1,1], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    info = r'$\ variance=%.2f$' %(np.var(sfconv_q_data))
    axes[1,1].set_xlabel(info)

    sns.barplot(x=list(range(1,len(var_ff)+1)), y=var_ff, color="limegreen", ax =axes[1,2] )
    axes[1,2].set_ylabel('Explained variance (%)')
    axes[1,2].set_xlabel('Number of non-zero SVs')
    for ind, label in enumerate(axes[1,2].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif (ind+1) % 8 == 0:   # every 4th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    
    fig.savefig('/data/pycode/SFConv/imgs/CT_weight.png', dpi=300, bbox_inches='tight')

def main():
    #vis_auroc()
    vis_weight()

if __name__ == '__main__':
    main()