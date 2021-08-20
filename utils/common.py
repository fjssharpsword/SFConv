# encoding: utf-8
"""
Training implementation for CIFAR10 dataset  
Author: Jason.Fang
Update time: 08/07/2021
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
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def count_bytes(file_size):
    '''
    Count the number of parameters in model
    '''
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def strofsize(integer, remainder, level):
        if integer >= 1024:
            remainder = integer % 1024
            integer //= 1024
            level += 1
            return strofsize(integer, remainder, level)
        else:
            return integer, remainder, level
    
    def MBofstrsize(integer, remainder, level):
        remainder = integer % (1024*1024)
        integer //= (1024*1024)
        level = 2
        return integer, remainder, level

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    #integer, remainder, level = strofsize(int(file_size), 0, 0)
    #if level+1 > len(units):
    #    level = -1
    integer, remainder, level = MBofstrsize(int(file_size), 0, 0)
    return ( '{}.{:>03d} {}'.format(integer, remainder, units[level]) )

def compute_AUCs(gt, pred, N_CLASSES):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def compute_ROCCurve(gt, pred, N_CLASSES, CLASS_NAMES, dataset_name):
    #fpr = 1-Specificity, tpr=Sensitivity
    np.set_printoptions(suppress=True) #to float
    thresholds = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    #color ref: https://www.cnblogs.com/darkknightzh/p/6117528.html
    color_name =['r','b','k','y','c','g','m','tan','gold','gray','coral','peru','lime','plum','seagreen']
    for i in range(N_CLASSES):
        fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}-AUROC{:.4f}'.format(CLASS_NAMES[i],auc_score))
        #select the prediction threshold
        #idx = np.where(tpr>auc_score)[0][0]
        #thresholds.append(threshold[idx])

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
    plt.savefig('/data/pycode/SFConv/imgs' + dataset_name +'_ROCCurve.jpg')

    return thresholds
    
def compute_fusion(gt, pred):
    #pred = F.log_softmax(pred, dim=1) 
    #pred = pred.max(1,keepdim=True)[1]
    gt_np = gt.cpu().numpy()[:,1] #positive
    pred_np = pred.cpu().numpy()[:,1]
    fpr, tpr, threshold = roc_curve(gt_np, pred_np)
    auc_score = auc(fpr, tpr)
    idx = np.where(tpr>auc_score)[0][0]#select the prediction threshold
    pred_np = np.where(pred_np>threshold[idx], 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    return sen, spe

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def dice_coeff(input, target):

    N = target.size(0)
    smooth = 1
    
    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)
    
    intersection = input_flat * target_flat
    
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    dice = loss.sum() / N

    return dice


def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        imageio.imwrite(str(index)+".png", feature_map[index-1])
    plt.savefig('/data/pycode/LungCT3D/imgs/fea_map1.jpg')

def transparent_back(img, gt=True):
    #img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0)) #alpha channel: 0~255
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
            else: 
                if gt: #true mask
                    color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                    img.putpixel(dot,color_1)
                else: #pred mask
                    color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                    img.putpixel(dot,color_1)
    return img