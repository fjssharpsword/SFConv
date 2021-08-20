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
#sys.path.append("..") 

def vis_seg_loss():
    """
    conv_loss = plt.imread('/data/pycode/SFConv/imgs/dice_conv.png')
    ffconv_loss = plt.imread('/data/pycode/SFConv/imgs/dice_ffconv.png') #rank-scale =0.1
    sfconv_loss = plt.imread('/data/pycode/SFConv/imgs/dice_sfconv.png') #rank-scale =0.1
    fig, axes = plt.subplots(1, 3, constrained_layout=True) 
    axes[0].imshow(conv_loss)
    axes[0].axis('off')
    axes[1].imshow(ffconv_loss)
    axes[1].axis('off')
    axes[2].imshow(sfconv_loss)
    axes[2].axis('off')
    fig.savefig('/data/pycode/SFConv/imgs/dice_loss.png', dpi=300, bbox_inches='tight')
    """
    datas = pd.read_csv('/data/pycode/SFConv/imgs/fundus_seg_diceloss.csv', sep=',')
    print(datas.shape)

def vis_seg_performance():

    rank_scale =  [0.1, 0.125, 0.25, 0.5, 1.0] #x_axies
    x_axies = [0, 1, 2, 3, 4]
    #conv 
    conv_dice = [0.9228, 0.9228, 0.9228, 0.9228, 0.9228] #y_axies
    conv_param = [16.49, 16.49, 16.49, 16.49, 16.49]
    conv_fps = [9.30, 9.30, 9.30, 9.30, 9.30]
    #FFConv
    ffconv_dice = [0.9345, 0.9435, 0.9250, 0.9508, 0.9126] #y_axies
    ffconv_param = [0.10, 1.20, 2.40, 4.78, 9.51]
    ffconv_fps = [9.77, 10.03, 10.10, 10.34, 10.43]

    #SFConv
    sfconv_dice = [0.9445, 0.9, 0.9, 0.9, 0.9] #y_axies
    sfconv_param = [0.10, 1.20, 2.40, 4.78, 9.51]
    sfconv_fps = [9.57, 10.09, 10.22, 10.28, 10.35]

    fig, axes = plt.subplots(3, 1, constrained_layout=True) #figsize=(10,18)
    #fig.suptitle('Performance and efficiency of segmentation for the Fundus dataset')
    axes[0].plot(x_axies, conv_dice,'bo-',label='Conv')
    axes[0].text(x_axies[0], conv_dice[0], conv_dice[0], ha='left', va='bottom', color='b') #
    axes[0].plot(x_axies, ffconv_dice,'g+-',label='FFConv')
    #for a, b in zip(x_axies, ffconv_dice):
    #    axes[0].text(a, b, b, ha='center', va='top', color='g')
    axes[0].text(x_axies[0], ffconv_dice[0], ffconv_dice[0], ha='left', va='bottom', color='g')
    axes[0].text(x_axies[1], ffconv_dice[1], ffconv_dice[1], ha='left', va='bottom', color='g')
    axes[0].text(x_axies[2], ffconv_dice[2], ffconv_dice[2], ha='left', va='bottom', color='g')
    axes[0].text(x_axies[3], ffconv_dice[3], ffconv_dice[3], ha='left', va='top', color='g')
    axes[0].text(x_axies[4], ffconv_dice[4], ffconv_dice[4], ha='right', va='bottom', color='g')
    axes[0].plot(x_axies, sfconv_dice,'r^-',label='SFConv(Ours)')
    axes[0].text(x_axies[0], sfconv_dice[0], sfconv_dice[0], ha='left', va='bottom', color='r')
    axes[0].text(x_axies[1], sfconv_dice[1], sfconv_dice[1], ha='left', va='bottom', color='r')
    axes[0].text(x_axies[2], sfconv_dice[2], sfconv_dice[2], ha='left', va='bottom', color='r')
    axes[0].text(x_axies[3], sfconv_dice[3], sfconv_dice[3], ha='left', va='bottom', color='r')
    axes[0].text(x_axies[4], sfconv_dice[4], sfconv_dice[4], ha='right', va='bottom', color='r')
    #for a, b in zip(x_axies, sfconv_dice):
    #    axes[0].text(a, b, b, ha='center', va='top', color='r')
    #axes[0].plot(x_axies, conv_dice,'bo-',x_axies, ffconv_dice,'g+-',x_axies, sfconv_dice,'r^-')
    axes[0].set_xticks([0, 1, 2, 3, 4])
    axes[0].set_xticklabels(['0.1', '0.125', '0.25', '0.5', '1.0'])
    #axes[0].set_xlabel('Rank scale')
    axes[0].set_ylabel('Dice coefficient')
    #axes[0].set_title('Model performance')
    axes[0].legend()

    axes[1].plot(x_axies, conv_param,'bo--',label='Conv')
    axes[1].text(x_axies[-1], conv_param[-1], conv_param[-1], ha='right', va='top')
    axes[1].plot(x_axies, ffconv_param,'g+-.',label='FFConv')
    for a, b in zip(x_axies, ffconv_param):
        axes[1].text(a, b, b, ha='center', va='bottom')
    axes[1].plot(x_axies, sfconv_param,'r^--',label='SFConv(Ours)')
    for a, b in zip(x_axies, sfconv_param):
        axes[1].text(a, b, b, ha='center', va='bottom')
    #axes[1].plot(x_axies, conv_param,'bo-',x_axies, ffconv_param,'g+-',x_axies, sfconv_param,'r^-')
    axes[1].set_xticks([0, 1, 2, 3, 4])
    axes[1].set_xticklabels(['0.1', '0.125', '0.25', '0.5', '1.0'])
    #axes[1].set_xlabel('Rank scale')
    axes[1].set_ylabel('Parameters(MB)')
    #axes[1].set_title('Model complexity')
    axes[1].legend()

    """
    ax_double = axes[1].twinx()
    ax_double.set_ylabel('Frams per second(FPS)')
    ax_double.plot(x_axies, conv_fps,'bo-',label='Conv')
    ax_double.plot(x_axies, ffconv_fps,'g+-',label='FFConv')
    ax_double.plot(x_axies, sfconv_fps,'r^-',label='SFConv(Ours)')
    #ax_double.plot(x_axies, conv_fps,'bo--',x_axies, ffconv_fps,'g+--',x_axies, sfconv_fps,'r^--')
    ax_double.legend()
    """

    axes[2].plot(x_axies, conv_fps,'bo-',label='Conv')
    axes[2].text(x_axies[0], conv_fps[0], conv_fps[0], ha='left', va='bottom', color='b')
    axes[2].plot(x_axies, ffconv_fps,'g+-',label='FFConv')
    #for a, b in zip(x_axies, ffconv_fps):
    #    axes[2].text(a, b, b, ha='right', va='bottom', color='g')
    axes[2].text(x_axies[0], ffconv_fps[0], ffconv_fps[0], ha='left', va='bottom', color='g')
    axes[2].text(x_axies[1], ffconv_fps[1], ffconv_fps[1], ha='left', va='top', color='g')
    axes[2].text(x_axies[2], ffconv_fps[2], ffconv_fps[2], ha='left', va='top', color='g')
    axes[2].text(x_axies[3], ffconv_fps[3], ffconv_fps[3], ha='left', va='bottom', color='g')
    axes[2].text(x_axies[4], ffconv_fps[4], ffconv_fps[4], ha='right', va='bottom', color='g')
    axes[2].plot(x_axies, sfconv_fps,'r^-',label='SFConv(Ours)')
    axes[2].text(x_axies[0], sfconv_fps[0], sfconv_fps[0], ha='left', va='top', color='r')
    axes[2].text(x_axies[1], sfconv_fps[1], sfconv_fps[1], ha='right', va='bottom', color='r')
    axes[2].text(x_axies[2], sfconv_fps[2], sfconv_fps[2], ha='right', va='bottom', color='r')
    axes[2].text(x_axies[3], sfconv_fps[3], sfconv_fps[3], ha='right', va='top', color='r')
    axes[2].text(x_axies[4], sfconv_fps[4], sfconv_fps[4], ha='right', va='top', color='r')
    #for a, b in zip(x_axies, sfconv_fps):
    #    axes[2].text(a, b, b, ha='left', va='top', color='r')
    axes[2].set_xticks([0, 1, 2, 3, 4])
    axes[2].set_xticklabels(['0.1', '0.125', '0.25', '0.5', '1.0'])
    axes[2].set_xlabel('Rank scale')
    axes[2].set_ylabel('Frams per second(FPS)')
    #axes[2].set_title('Speed of model inference')
    axes[2].legend()

    fig.savefig('/data/pycode/SFConv/imgs/fundus_seg.png', dpi=300, bbox_inches='tight')


def main():
    #vis_seg_performance()
    vis_seg_loss()
    


if __name__ == '__main__':
    main()