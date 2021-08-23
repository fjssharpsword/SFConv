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
Dataset: Medical MNIST, 58954 medical images of 6 classes
https://www.kaggle.com/andrewmvd/medical-mnist
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
        data = pd.read_csv(path_to_dataset_file, sep=',')
        self.CLASS_NAMES = ['AbdomenCT', 'BreastMRI', 'Hand', 'CXR', 'HeadCT', 'ChestCT']
        data = data.values #dataframe -> numpy
        image_list = []
        label_list = []
        for rec in data:
            image_list.append(rec[0])
            label_list.append([int(rec[1])])
        self.image_list = image_list
        self.label_list = label_list
        
    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.ToTensor()])
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
        label = torch.as_tensor(label, dtype=torch.long)
        
        return image, label

    def __len__(self):
        return len(self.image_list)

def get_dataloader_train(batch_size, shuffle, num_workers):
    csv_file = '/data/pycode/SFConv/dsts/medmnist_train.txt'
    image_dir = '/data/fjsdata/MedMNIST/'
    dataset_box = DatasetGenerator(path_to_img_dir=image_dir, path_to_dataset_file=csv_file)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box

def get_dataloader_test(batch_size, shuffle, num_workers):
    csv_file = '/data/pycode/SFConv/dsts/medmnist_test.txt'
    image_dir = '/data/fjsdata/MedMNIST/'
    dataset_box = DatasetGenerator(path_to_img_dir=image_dir, path_to_dataset_file=csv_file)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box

def split_medical_minist():
    root='/data/fjsdata/MedMNIST/'
    class_name = ['AbdomenCT', 'BreastMRI', 'Hand', 'CXR', 'HeadCT', 'ChestCT']
    names = []
    labels = []
    for _, dirs, _ in os.walk(root):
        for dir in dirs:
            dir =os.path.join(root,dir)
            label_name = dir.split('/')[-1]
            label_idx = class_name.index(label_name)
            for file in os.listdir(dir):
                names.append(os.path.join(dir,file))
                labels.append(label_idx)
    #train-test
    X_train, X_test, y_train, y_test = train_test_split(np.array(names), np.array(labels), test_size=0.33, random_state=42)
    train_set = pd.DataFrame({'name': X_train, 'label': y_train})
    test_set = pd.DataFrame({'name': X_test, 'label': y_test})
    print("\r distribution of train_set: {}".format(train_set['label'].value_counts())) 
    print("\r distribution of test_set: {}".format(test_set['label'].value_counts())) 
    train_set.to_csv('/data/pycode/SFConv/dsts/medmnist_train.txt', index=False, sep=',')
    test_set.to_csv('/data/pycode/SFConv/dsts/medmnist_test.txt', index=False, sep=',')
    #trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/pycode/SFConv/dsts/medmnist_train.txt', index=False, sep=',')
    #testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/pycode/SFConv/dsts/medmnist_test.txt', index=False, sep=',')

if __name__ == "__main__":

    #split_medical_minist()

    #for debug   
    data_loader_box = get_dataloader_test(batch_size=8, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader_box):
        print(len(image))
        print(len(label))
        break
