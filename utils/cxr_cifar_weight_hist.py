# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 28/06/2022
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

def vis_weight():

    #load data
    img_type =['Natural images', 'Medical images']
    nat_path = '/data/pycode/SFConv/log/cifar/'
    med_path = '/data/pycode/SFConv/log/cxr/'

    nat_list_s,  nat_list_e= [], []
    for root, ds, fs in os.walk(nat_path):
        for f in fs:
            nat_npy =  np.load(nat_path + f).flatten()
            if 's' in f:
                nat_list_s.extend(nat_npy)
            else: 
                nat_list_e.extend(nat_npy)
    med_list_s,  med_list_e= [], []
    for root, ds, fs in os.walk(med_path):
        for f in fs:
            med_npy =  np.load(med_path + f).flatten()
            if 's' in f:
                med_list_s.extend(med_npy)
            else: 
                med_list_e.extend(med_npy)

    nat_s_data =  np.array(nat_list_s).flatten()
    nat_e_data =  np.array(nat_list_e).flatten()
    med_s_data =  np.array(med_list_s).flatten()
    med_e_data =  np.array(med_list_e).flatten()

    """
    #plot 
    fig, axes = plt.subplots(1,2,constrained_layout=True)#figsize=(10,5)

    #natural
    axes[0].set_ylabel('Value')
    axes[0].set_xlabel('Number')
    axes[0].set_title(img_type[0])
    info_s = r'$\ variance=%.2f$' %(np.var(nat_s_data))
    sns.distplot(nat_s_data, kde=True, ax=axes[0], hist_kws={'color':'red'}, kde_kws={'color':'green'}, label=info_s)
    info_e = r'$\ variance=%.2f$' %(np.var(nat_e_data))
    sns.distplot(nat_e_data, kde=True, ax=axes[0], hist_kws={'color':'red'}, kde_kws={'color':'blue'}, label=info_e)
    #axes[0].legend()

    #medical
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Number')
    axes[1].set_title(img_type[1])
    info_s = r'$\ variance=%.2f$' %(np.var(med_s_data))
    sns.distplot(med_s_data, kde=True, ax=axes[1], hist_kws={'color':'green'}, kde_kws={'color':'red'}, label=info_s)
    info_e = r'$\ variance=%.2f$' %(np.var(med_e_data))
    sns.distplot(med_e_data, kde=True, ax=axes[1], hist_kws={'color':'blue'}, kde_kws={'color':'red'}, label=info_e)
    #axes[1].legend()
    """
    fig, axe = plt.subplots(1)
    axe.set_ylabel('Number')
    axe.set_xlabel('Value')
    sns.distplot(nat_s_data, kde=True, ax=axe, hist_kws={'color':'green'}, kde_kws={'color':'green'}, label="Pre-train")
    sns.distplot(nat_e_data, kde=True, ax=axe, hist_kws={'color':'blue'}, kde_kws={'color':'blue'}, label="Post-train (natural images)")
    sns.distplot(med_e_data, kde=True, ax=axe, hist_kws={'color':'red'}, kde_kws={'color':'red'}, label="Post-train (medical images)")
    axe.legend()

    #output    
    fig.savefig('/data/pycode/SFConv/imgs/cxr_cifar_weight_hist.png', dpi=300, bbox_inches='tight')

def main():
    vis_weight()

if __name__ == '__main__':
    main()