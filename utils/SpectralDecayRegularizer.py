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
    fig, axes = plt.subplots(2,2,constrained_layout=True)
    #FFConv
    axes[0,0].set_ylabel('$\ L_{2}$')
    info = r'$\ \sigma=%.2f$' %(np.std(ffconv_w_data)*100)
    axes[0,0].set_title(info)
    sns.distplot(ffconv_w_data, kde=True, ax=axes[0,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    axes[0,0].grid()

    sns.barplot(x=list(range(1,len(var_ff)+1)), y=var_ff*100, color="limegreen", ax =axes[0,1] )
    axes[0,1].set_title(r'$\ EVR@SN=%.2f$' %(var_ff[0]*100))
    for ind, label in enumerate(axes[0,1].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif (ind+1) % 8 == 0:  # every 4th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    axes[0,1].grid(axis='y')

    #SFConv
    axes[1,0].set_ylabel('SD (Ours)')
    info = r'$\ \sigma=%.2f$' %(np.std(sfconv_w_data)*100)
    axes[1,0].set_title(info)
    sns.distplot(sfconv_w_data, kde=True, ax=axes[1,0], hist_kws={'color':'green'}, kde_kws={'color':'red'})
    axes[1,0].grid()
    
    sns.barplot(x=list(range(1,len(var_sf)+1)), y=var_sf*100, color="limegreen", ax =axes[1,1] )
    axes[1,1].set_title(r'$\ EVR@SN=%.2f$' %(var_sf[0]*100))
    for ind, label in enumerate(axes[1,1].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif (ind+1) % 8 == 0:   # every 4th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    axes[1,1].grid(axis='y')
    
    fig.savefig('/data/pycode/SFConv/imgs/SDReg_Conv_Var.png', dpi=300, bbox_inches='tight')

def main():
    vis_weight()

if __name__ == '__main__':
    main()