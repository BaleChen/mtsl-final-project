import os
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

import pandas as pd
import numpy as np

from UniformAugment import UniformAugment

#########################
##   Dataset Classes   ##
#########################

class SeedlingDataset(Dataset):
    def __init__(self, df, path, small_sample=False, transform=None, use_train_folder=True):
        self.df = df
        self.path = path
        self.transform = transform.transforms.insert(0, UniformAugment())
        if small_sample:
            self.df = self.df.sample(50)
        if use_train_folder:
            self.path += "train/"
        else:
            self.path += "test/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = Image.open(self.path + img_path).convert("RGB")
        label = self.df.iloc[idx, 2]

        if self.transform:
            img = self.transform(img)

        return img, label

#########################
##  Data Augmentation  ##
#########################

class IdentityTransform(object):
    
    def __call__(self, data):
        return data
    
class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    def __init__(self, alphastd, eigval=torch.Tensor([0.2175, 0.0188, 0.0045]), eigvec=torch.Tensor([[-0.5675,  0.7192,  0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948,  0.4203]])):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def get_train_trans(image_size=224, data_aug = False):
    """
    Transform function for processing images in the training set.
    """
    if data_aug:
        return transforms.Compose([
            transforms.RandomCrop(image_size, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomRotation(degrees=(60, 90)),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
            #transforms.ColorJitter(brightness=0.1, saturation=0.1),
            Lighting(0.9),
            transforms.RandomErasing(scale=(0.02, 0.05), ratio=(0.7, 0.9)),
        ])
    else: 
        return transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
def get_val_trans(image_size=224):
    """
    Transform function for processing images in the validation/test set.
    """
    return transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    

#################
##  Functions  ##
#################

def get_classes_indices_mapping(data_dir):
    """
    Input the data folder directory.
    
    Return list of classes, mapping from class name to index, and backwards.
    """
    classes = sorted([d for d in os.listdir(data_dir + "train/")])
    class_to_idx = {cls:i for i, cls in enumerate(classes)}
    idx_to_class = {i:cls for i, cls in enumerate(classes)}
    
    return classes, class_to_idx, idx_to_class


def get_data_df(data_dir):
    """
    Input the data folder directory.
    
    Return a pandas DataFrame of the image paths, their corresponding labels, and label indices.
    """
    classes, class_to_idx, idx_to_class = get_classes_indices_mapping(data_dir)
    
    data_l = []
    for idx, label in enumerate(classes):
        path_to_image = data_dir + "train/" + label
        for file in listdir(path_to_image):
            data_l.append([f"{label}/{file}", label, idx])

    data_df = pd.DataFrame(data_l, columns=["filename", "label", "index"])
    
    return data_df

def get_train_val_loader(data_dir, val_ratio=0.2, train_trans=None, val_trans=None,
                        batch_size=32, small_sample=False):
    """
    Generate the train and validation dataloaders.
    """
    
    train_df, val_df = train_test_split(get_data_df(data_dir), test_size=val_ratio)
    
    train_set = SeedlingDataset(train_df, data_dir, small_sample=small_sample,
                                transform = train_trans)
    val_set = SeedlingDataset(val_df, data_dir, small_sample=small_sample,
                              transform = val_trans)
    
    return ( DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=True) )
    
def test_image_loader(im_dir, trans):
    """
    Image loader for each test set image
    """
    im = Image.open(im_dir).convert("RGB")
    im = trans(im)
    im = im.unsqueeze(0)
    return im
