import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import re
import sys
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
from pycocotools import mask as coco_mask
import pickle
from sklearn.model_selection import train_test_split
"""
Dataset: Normal=0, Pneumonia=1, and COVID-19=2
https://www.kaggle.com/hgunraj/covidxct
"""
#generate 
#https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#https://github.com/pytorch/vision
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        data = pd.read_csv(path_to_dataset_file, sep=" ", header=None)
        data.columns=['filename', 'label', 'xmin','ymin','xmax','ymax']
        data=data.drop(['xmin', 'ymin','xmax', 'ymax'], axis=1 )

        self.labels={0:'Normal',1:'Pneumonia',2:'COVID19'}
        self.class_names=['Normal','Pneumonia','COVID19']
        data = data.values #dataframe -> numpy
        image_list = []
        label_list = []
        for rec in data:
            image_list.append(os.path.join(path_to_img_dir,rec[0]))
            onehot_lbl = np.zeros(len(self.class_names))
            onehot_lbl[int(rec[1])] = 1
            label_list.append(onehot_lbl)

        self.image_list = image_list
        self.label_list = label_list
           
    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        return transform_seq(img)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        #image
        image = self.image_list[index]
        image = Image.open(image)
        image = self._transform_tensor(image)
        #label
        label = self.label_list[index]
        label = torch.as_tensor(label, dtype=torch.float32)
        
        return image, label

    def __len__(self):
        return len(self.image_list)

def get_dataloader_train(batch_size, shuffle, num_workers):
    csv_file = '/data/fjsdata/COVIDxCT/train_COVIDx_CT-2A.txt'
    image_dir = '/data/fjsdata/COVIDxCT/2A_images/'
    dataset_box = DatasetGenerator(path_to_img_dir=image_dir, path_to_dataset_file=csv_file)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box

def get_dataloader_val(batch_size, shuffle, num_workers):
    csv_file = '/data/fjsdata/COVIDxCT/val_COVIDx_CT-2A.txt'
    image_dir = '/data/fjsdata/COVIDxCT/2A_images/'
    dataset_box = DatasetGenerator(path_to_img_dir=image_dir, path_to_dataset_file=csv_file)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box

def get_dataloader_test(batch_size, shuffle, num_workers):
    csv_file = '/data/fjsdata/COVIDxCT/test_COVIDx_CT-2A.txt'
    image_dir = '/data/fjsdata/COVIDxCT/2A_images/'
    dataset_box = DatasetGenerator(path_to_img_dir=image_dir, path_to_dataset_file=csv_file)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box


if __name__ == "__main__":
    #for debug   
    data_loader = get_dataloader_val(batch_size=8, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader):
        print(len(image))
        print(len(label))
        break
