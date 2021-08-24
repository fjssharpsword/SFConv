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

    class_names=['Pneumonia','COVID19']

    #fpr = 1-Specificity, tpr=Sensitivity
    np.set_printoptions(suppress=True) #to float
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    #color ref: https://www.cnblogs.com/darkknightzh/p/6117528.html
    color_name =['r','b','k','y','c','g','m']
    for i in range(N_CLASSES):
        fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}-AUROC{:.4f}'.format(CLASS_NAMES[i],auc_score))

    #plot and save
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right')
    #plt.title('ROC curve')
    plt.savefig('/data/pycode/SFConv/imgs/CT_ROCCurve.jpg')

    return thresholds


def main():
    vis_auroc()

if __name__ == '__main__':
    main()