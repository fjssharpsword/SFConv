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

def vis_cxr_data():
 
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

def vis_cxr_lesion():
    """
    print('********************load data********************')
    cxr_set = get_box_dataloader_VIN(batch_size=1, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('*******Calcluation*********')
    skews = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[]}
    for batch_idx, (images, targets) in enumerate(cxr_set):
        for i in range(len(targets[0]['labels'])):
            #data
            cxr_box = targets[0]['boxes'].numpy()[i]
            cxr_box = images[0].numpy()[:,int(cxr_box[0]):int(cxr_box[2]),int(cxr_box[1]):int(cxr_box[3])]
            cxr_box = cxr_box.flatten()
            try:
                [_, _, box_skew, _] = calc_stat(cxr_box)
            except:
                continue
            box_lbl = targets[0]['labels'].numpy()[i] #label
            skews[box_lbl].append(abs(box_skew))

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    print('*******Plot*********')
    skews_avg = []
    for key in skews.keys():
        skews_avg.append(np.mean(skews[key]))
    print(skews_avg)
    """
    skews_avg = [0.6647851184382049, 1.2134887321210696, 1.0870273371157367, 0.895383005946241, 0.5418230802887915, 1.3881208990471322, \
                0.7471606403874295, 0.7374767619287942, 0.9879827125356338, 0.6228555838749348, 0.7489556802321962, 0.9393112767901272, \
                1.2605938126261683, 0.6262070169396955]
    class_name = ['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'Interstitial lung disease', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    data = {'Lesion type': class_name, 'Average skewness': skews_avg}
    data = pd.DataFrame(data)
    fig, ax = plt.subplots(1) #figsize=(6,9)
    ax = sns.barplot(x="Lesion type", y="Average skewness", data=data)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(90)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    ax.hlines(np.mean(skews_avg), 0, 13, colors = "r", linestyles = "dashed")
    ax.set_ylim(0.4,1.4)
    #ax.set_yticklabels(['0.0', '5.0', '10.0', '15.0'])
    fig.savefig('/data/pycode/SFConv/imgs/lesion_dis.png', dpi=300, bbox_inches='tight')

def vis_cxr_lesion_new():
    skews_avg = [0.6647851184382049, 1.2134887321210696, 1.0870273371157367, 0.895383005946241, 0.5418230802887915, 1.3881208990471322, \
                0.7471606403874295, 0.7374767619287942, 0.9879827125356338, 0.6228555838749348, 0.7489556802321962, 0.9393112767901272, \
                1.2605938126261683, 0.6262070169396955]
    class_name = ['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'Interstitial lung disease', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    map_dif_resnet = [0.9736-0.9869, 0.4333-0.2667, 0.2527-0.1429, 0.9971-0.9971, 0.7159-0.7563, 0.8826-0.6854, 0.6284-0.7701, 0.6124-0.7074,\
                      0.3738-0.2729, 0.3496-0.3602, 0.8465-0.8445, 0.7474-0.6806, 0.6222-0.4222, 0.5271-0.5668]
    map_dif_densenet = [0.9759-0.9959, 0.4833-0.2333, 0.1868-0.1209, 0.9962-0.9957, 0.7311-0.7479, 0.8310-0.6103, 0.5824-0.7011, 0.6066-0.6550,\
                        0.3776-0.3028, 0.2606-0.3284, 0.8150-0.8012, 0.7610-0.6388, 0.6000-0.3778, 0.5162-0.5535]
    data_res = {'Lesion type': class_name, 'Average skewness': skews_avg, 'Difference of average precision': map_dif_resnet, 'Model type': 'ResNet18'}
    data_des = {'Lesion type': class_name, 'Average skewness': skews_avg, 'Difference of average precision': map_dif_densenet, 'Model type': 'DenseNet121'}  
    data_res = pd.DataFrame(data_res)
    data_des = pd.DataFrame(data_des)
    data = pd.concat([data_res, data_des],axis=0)

    fig, ax = plt.subplots(1) #figsize=(6,9)
    ax = sns.scatterplot(data=data, x="Average skewness", y="Difference of average precision", hue="Lesion type", style='Model type', sizes=(20, 200))
    #ax = sns.relplot(x="Average skewness", y="Difference of average precision", hue="Lesion type", style='Model type', size="Lesion type",sizes=(20, 200), data=data) 
    ax.grid()
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=3)
    fig.savefig('/data/pycode/SFConv/imgs/lesion_dis_new.png', dpi=300, bbox_inches='tight')

def main():
    #vis_cxr_data()
    #vis_cxr_lesion()
    vis_cxr_lesion_new()

if __name__ == '__main__':
    main()