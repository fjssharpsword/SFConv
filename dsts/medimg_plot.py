# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 06/09/2021
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
from vincxr_det import get_box_dataloader_VIN
from fundus_seg import get_test_dataloader
from COVIDx_ct import get_dataloader_test

def vis_med_img():

    fig, axes = plt.subplots(1,3,constrained_layout=True, figsize=(9,3))

    print('********************Fundus image********************')
    fundus_data = get_test_dataloader(batch_size=1, shuffle=False, num_workers=0) 
    with torch.autograd.no_grad():
        for batch_idx, (image, mask) in enumerate(fundus_data):
                img = image.squeeze().numpy().transpose(1,2,0)
                #msk = mask.numpy().repeat(3, axis=0).transpose(1,2,0)
                #img =  img + msk
                #img = np.where(img>1, 1, img)
                axes[0].imshow(img, aspect="auto")
                axes[0].axis('off')
                axes[0].set_title('Fundus for segmentation')
                #axes[0].set_xlabel('Segmentation of the optic disc')
                break

    print('********************Chest X-ray********************')
    cxr_data = get_box_dataloader_VIN(batch_size=1, shuffle=False, num_workers=0)
    CLASS_NAMES_Vin = ['No Finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(cxr_data):
            img = images[0]
            box = targets[0]['boxes'][0]
            lbl = targets[0]['labels'][0]
            #plot goundtruth box
            img = img.cpu().numpy().transpose(1,2,0)
            axes[1].imshow(img, aspect="auto")
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)# add groundtruth
            axes[1].text(box[0]-20, box[1]-5, CLASS_NAMES_Vin[lbl])
            axes[1].axis('off')
            axes[1].set_title('Chest X-ray for detection')
            #axes[1].set_xlabel('Abnormalities detection')
            break

    print('********************CT********************')
    ct_data = get_dataloader_test(batch_size=1, shuffle=False, num_workers=0)
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(ct_data):
            if list(lbl[0].numpy()).index(1) == 2: #COVID19
                img = img.squeeze(0).numpy().transpose(1,2,0)
                axes[2].imshow(img, aspect="auto",cmap='gray')
                axes[2].axis('off')
                #axes[2].get_xaxis().set_visible(False)
                #axes[2].get_yaxis().set_visible(False)
                #axes[2].set_title('CT image \n Classification of pneumonia and COVID19')
                axes[2].set_title('CT for classification')
                #axes[2].set_xlabel('Classification of pneumonia and COVID19')
                break
    fig.savefig('/data/pycode/SFConv/imgs/med_img.png', dpi=300, bbox_inches='tight')


    """
    fig, axes = plt.subplots(1,3,constrained_layout=True) #figsize=(9,3)

    fundus = Image.open('/data/pycode/SFConv/imgs/ctpred/fundus_seg.jpg')
    axes[0].imshow(fundus, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('Fundus')

    cxr = Image.open('/data/pycode/SFConv/imgs/ctpred/cxr_det.jpg')
    axes[1].imshow(cxr,aspect="auto")
    axes[1].axis('off')
    axes[1].set_title('Chest X-ray')

    ct = Image.open('/data/pycode/SFConv/imgs/ctpred/ct.png')
    axes[2].imshow(ct,cmap='gray',aspect="auto")
    axes[2].axis('off')
    axes[2].set_title('CT')

    fig.savefig('/data/pycode/SFConv/imgs/med_img.png', dpi=300, bbox_inches='tight')
    """

def vis_cxr_det():
    print('********************Chest X-ray********************')
    cxr_data = get_box_dataloader_VIN(batch_size=1, shuffle=False, num_workers=0)
    CLASS_NAMES_Vin = ['No Finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(cxr_data):
            img = images[0]
            box = targets[0]['boxes'][0]
            lbl = targets[0]['labels'][0]
            #plot goundtruth box
            img = img.cpu().numpy().transpose(1,2,0)

            fig, axes = plt.subplots(1,2,constrained_layout=True, figsize=(6,3))

            axes[0].imshow(img, aspect="auto")
            rect_t = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect_t)# add groundtruth
            rect_p = patches.Rectangle((box[0]+6, box[1]-3), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
            axes[0].add_patch(rect_p)# add groundtruth
            axes[0].text(box[0]-20, box[1]-5, CLASS_NAMES_Vin[lbl])
            axes[0].axis('off')
            axes[0].set_title('Conv(norm CNN)')

            axes[1].imshow(img, aspect="auto")
            rect_t = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect_t)# add groundtruth
            rect_p = patches.Rectangle((box[0]+3, box[1]+3), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
            axes[1].add_patch(rect_p)# add groundtruth
            axes[1].text(box[0]-20, box[1]-5, CLASS_NAMES_Vin[lbl])
            axes[1].axis('off')
            axes[1].set_title('SFConv(Ours)')

            fig.savefig('/data/pycode/SFConv/imgs/det_cxr_vis.png', dpi=300, bbox_inches='tight')
            break

def main():
    #vis_med_img()
    vis_cxr_det()

if __name__ == '__main__':
    main()