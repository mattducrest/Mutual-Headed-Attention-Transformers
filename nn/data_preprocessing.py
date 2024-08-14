#necessary imports
import os
import timm
import h5py
import numpy as np
import pandas as pd
import random as rd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import OrdinalEncoder

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from albumentations.pytorch import ToTensorV2
import albumentations as A

np.random.seed(42)
torch.manual_seed(42)

#image loader class for h5py
class ImageLoaderWithMetadata(Dataset):
    def __init__(self, df, file_hdf, transform=None, subset_size=None, has_target=True):
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.transform = transform
        self.has_target = has_target
        self.isic_ids = df['isic_id'].tolist()
        
        if subset_size is not None and subset_size < len(df):
            self.df = df.sample(n=subset_size).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        
        if self.has_target:
            self.targets = self.df['target'].values
            self.metadata_cols = self.df.drop(columns=['target', 'isic_id']).columns
        else:
            self.metadata_cols = self.df.drop(columns=['isic_id']).columns

    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
        
        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed['image']
            image = image / 255 
        
        metadata = self.df.loc[index, self.metadata_cols].values.astype(np.float32)
        
        if self.has_target:
            target = self.targets[index]
            return (image, metadata, target)
        else:
            return image, metadata

#image loader class for jpegs
class ImageLoaderWithMetadata_jpgs(Dataset):
    def __init__(self, df, image_path, has_target=True, transform=None, subset_size=None):
        self.df = df
        self.transform = transform
        self.has_target = has_target
        self.isic_ids = df['isic_id'].tolist()
        
        self.img_path = os.path.join(image_path)
        
        if subset_size is not None and subset_size < len(df):
            self.df = df.sample(n=subset_size).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        
        if self.has_target:
            self.targets = self.df['target'].values
            self.metadata = self.df.drop(columns=['target', 'isic_id']).values  # Ensure isic_id is dropped
        else:
            self.metadata = self.df.drop(columns=['isic_id']).values  # Ensure isic_id is dropped

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img_path = os.path.join(self.img_path, isic_id + '.jpg')  
        image = Image.open(img_path)
        
        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)  # Apply transformation
            image = transformed['image']
            image = image / 255 

        metadata = self.metadata[index]
        
        if self.has_target:
            target = self.targets[index]
            return (image, metadata, target)
        else:
            return image, metadata

#set image size and augmentations
image_size = (224,224)

train_transform_no_augment = A.Compose([
    
    A.Resize(image_size[0], image_size[1]),  # Resize to the target size
    ToTensorV2(),  # Convert the image to a PyTorch tensor
])

train_transfrom_with_augment = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.01, rotate_limit=20, p=0.5),
    A.HorizontalFlip(p=0.7),
    A.VerticalFlip(p=0.7),   
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.5, p=0.8),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.8),  # Slight R, G and B shift
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.8),  # Slight changes to hue, saturation, and value (brightness)
    
    A.Resize(image_size[0], image_size[1]),  # Resize to the target size
    ToTensorV2(),  # Convert the image to a PyTorch tensor
])

#metadata and target
class ConcatDatasetWithMetadataAndTarget(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets) 
        self.targets = np.concatenate([dataset.targets for dataset in datasets])
    
    def __getitem__(self, idx):
        image, metadata, target = super().__getitem__(idx)
        
        target = self.targets[idx]
        return image, metadata, target