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

def vis_med_img():
    fig, axes = plt.subplots(1,3,constrained_layout=True, figsize=(9,3))

    fundus = Image.open('/data/pycode/SFConv/imgs/ctpred/fundus.jpg')
    axes[0].imshow(fundus, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('Fundus')

    cxr = Image.open('/data/pycode/SFConv/imgs/ctpred/cxr.jpeg')
    axes[1].imshow(cxr,aspect="auto")
    axes[1].axis('off')
    axes[1].set_title('Chest X-ray')

    ct = Image.open('/data/pycode/SFConv/imgs/ctpred/ct.png')
    axes[2].imshow(ct,cmap='gray',aspect="auto")
    axes[2].axis('off')
    axes[2].set_title('CT')

    fig.savefig('/data/pycode/SFConv/imgs/med_img.png', dpi=300, bbox_inches='tight')

def main():
    vis_med_img()

if __name__ == '__main__':
    main()