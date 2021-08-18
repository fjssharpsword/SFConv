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
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
from dsts.vincxr_coco import get_box_dataloader_VIN

def main():
 
    print('********************load data********************')
    cxr_dst = get_box_dataloader_VIN(batch_size=1, shuffle=True, num_workers=0)
    test_set = dset.VOCSegmentation(root='/data/fjsdata/VOC2012/', year='2012', image_set='val', transform=trans, download=False) 
    voc_dst = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size= 1,
                    shuffle= True, num_workers=0)
    print('********************load data succeed!********************')

    print('*******Plot!*********')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):

            img = images[0]
            box = targets[0]['boxes'][0]
            lbl = targets[0]['labels'][0]
            #plot goundtruth box
            fig, ax = plt.subplots(1)# Create figure and axes
            img = img.cpu().numpy().transpose(1,2,0)
            ax.imshow(img)
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)# add groundtruth
            ax.text(box[0], box[1], CLASS_NAMES_Vin[lbl])
            ax.axis('off')
            fig.savefig('/data/pycode/SFConv/imgs/cxr_img.png', dpi=300, pad_inches=0, bbox_inches='tight')
            fig, ax = plt.subplots(1)
            ax = sns.kdeplot(images[0].numpy().flatten(), shade=True, color="g")
            fig.savefig('/data/pycode/SFConv/imgs/cxr_dis.png', dpi=300, pad_inches=0, bbox_inches='tight')

            """
            images = list(image.cuda() for image in images)
            targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
            fea_map = model.backbone(images[0].unsqueeze(0))#forward
            #log_writer.add_histogram('cxr_fea', fea_map, 2)
            plt.hist(fea_map.cpu().numpy().flatten(), facecolor='r', alpha=0.75, density=True, linewidth=0.5) #orange
            #ax = sns.distplot(fea_map.cpu().numpy().flatten())
            plt.savefig('/data/pycode/LungCT3D/imgs/fea_5.jpg')

            img = images[0]
            box = targets[0]['boxes'][0]
            lbl = targets[0]['labels'][0]
            #plot goundtruth box
            fig, ax = plt.subplots()# Create figure and axes
            img = img.cpu().numpy().transpose(1,2,0)
            ax.imshow(img)
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)# add groundtruth
            ax.text(box[0], box[1], CLASS_NAMES_Vin[lbl])
            ax.axis('off')
            fig.savefig('/data/pycode/LungCT3D/imgs/img_3.jpg')
            #log_writer.add_histogram('cxr_data', img, 2)
          
            fea_map = fea_map.cpu().numpy().squeeze()
            fea_map = np.mean(fea_map, axis=0)
            #fea_map = np.maximum(fea_map, 0)
            #fea_map /= np.max(fea_map)
            #plt.matshow(fea_map)
            #plt.savefig('/data/pycode/LungCT3D/imgs/fea_1.jpg')
            fea_map = cv2.resize(fea_map, (img.shape[0], img.shape[1]))  
            fea_map = np.uint8(255 * fea_map) 
            fea_map = cv2.applyColorMap(fea_map, cv2.COLORMAP_JET)  
            img = np.uint8(255 * img) 
            overlay_img = fea_map * 0.3 + img 
            cv2.imwrite('/data/pycode/LungCT3D/imgs/fea_1.jpg', overlay_img)
            """
            break
    #log_writer.close() #shut up the tensorboard


    def VisSeg():
        print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=1, shuffle=True, num_workers=0) #BATCH_SIZE
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = UNet(n_channels=3, n_classes=1).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Vin-CXR dataset: "+CKPT_PATH)
    model.eval()
    criterion = DiceLoss().cuda()
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    time_res = []
    dice_coe = []
    with torch.autograd.no_grad():
        for batch_idx, (image, mask) in enumerate(dataloader_test):

            #plot goundtruth box
            fig, ax = plt.subplots()# Create figure and axes
            img = image.squeeze().numpy().transpose(1,2,0)
            #msk = mask.numpy().repeat(3, axis=0).transpose(1,2,0)
            #img =  img + msk
            #img = np.where(img>1, 1, img)
            ax.imshow(img)
            fig.savefig('/data/pycode/SFConv/imgs/fundus_img.png', dpi=300, pad_inches=0, bbox_inches='tight')
            fig, ax = plt.subplots(1)
            ax = sns.kdeplot(img.flatten(), shade=True, color="g")
            fig.savefig('/data/pycode/SFConv/imgs/fundus_dis.png', dpi=300, pad_inches=0, bbox_inches='tight')

            """
            var_image = torch.autograd.Variable(image).cuda()
            var_out = model(var_image)
            #plot goundtruth box
            fig, ax = plt.subplots()# Create figure and axes
            img = image.squeeze().numpy().transpose(1,2,0)
            msk = mask.numpy().repeat(3, axis=0).transpose(1,2,0)
            prd = var_out.cpu().squeeze(0).numpy().repeat(3, axis=0).transpose(1,2,0)
            img =  img + msk*prd
            img = np.where(img>1, 1, img)
            ax.imshow(img)
            ax.axis('off')
            fig.savefig('/data/pycode/SFConv/imgs/fundus_conv.jpg')
            #img = np.uint8(255 * img) 
            #img = cv2.applyColorMap(img, cv2.COLORMAP_JET) 
            #msk = mask.numpy().squeeze()
            #msk = np.uint8(255 * msk) 
            #msk = cv2.applyColorMap(msk, cv2.COLORMAP_JET)  
            #overlay_img = msk * 0.3 + img 
            #cv2.imwrite('/data/pycode/LungCT3D/imgs/fundus.jpg', overlay_img)
            """
            break


        VisSeg()
    """
    fig, ax = plt.subplots()# Create figure and axes
    img = Image.open('/data/tmpexec/000001.jpg')
    ax.imshow(img)
    fig.savefig('/data/pycode/SFConv/imgs/imagenet_img.png', dpi=300, pad_inches=0, bbox_inches='tight')
    fig, ax = plt.subplots(1)
    ax = sns.kdeplot(np.asarray(img).flatten(), shade=True, color="g")
    fig.savefig('/data/pycode/SFConv/imgs/imagenet_dis.png', dpi=300, pad_inches=0, bbox_inches='tight')
    """


if __name__ == '__main__':
    main()