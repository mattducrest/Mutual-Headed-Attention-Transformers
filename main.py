from nn.data_preprocessing import *
from nn.model import *
from train.train import *

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

#load the data
train_metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
train_image_path = '/kaggle/input/isic-2024-challenge/train-image.hdf5'
isic_2024_metadata_df = pd.read_csv(train_metadata_path)

features_remove = ['lesion_id','attribution', 'copyright_license', 'image_type','iddx_full','iddx_1','iddx_2','iddx_3','iddx_4','iddx_5','mel_mitotic_index', 'mel_thick_mm','tbp_lv_dnn_lesion_confidence']

#set device to gpu if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#soft label encode the dataframe
soft_shuffled_df = soft_label_encoder(isic_2024_metadata_df)

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(soft_shuffled_df, test_size=0.2, random_state=42, stratify = soft_shuffled_df['target'])

# Create datasets Training Datasets (With and Without augmentation)
train_benign_dataset_no     = ImageLoaderWithMetadata(df=train_df[train_df['target'] == 0], file_hdf=train_image_path, transform=train_transform_no_augment)
train_malignant_dataset_no  = ImageLoaderWithMetadata(df=train_df[train_df['target'] == 1], file_hdf=train_image_path, transform=train_transform_no_augment)
train_malignant_dataset_aug = ImageLoaderWithMetadata(df=train_df[train_df['target'] == 1], file_hdf=train_image_path, transform=train_transfrom_with_augment)

# Concatenate both
train_dataset = ConcatDatasetWithMetadataAndTarget([train_benign_dataset_no, train_malignant_dataset_no, train_malignant_dataset_aug])

# Create Validation dataset (No Augmentation)
val_dataset   = ImageLoaderWithMetadata(df=val_df,   file_hdf=train_image_path, transform=train_transform_no_augment)

# Create DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=4)
val_loader   = DataLoader(val_dataset, shuffle=False, batch_size=64, num_workers=4)

"""Training"""

# Initialize the multimodal transformer
multimodal_transformer = Multimodal_Transformer()
multimodal_transformer.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = Adam(multimodal_transformer.parameters(), lr=0.00001)

# Training loop
num_epochs = 20

history = train_model(num_epochs, multimodal_transformer, train_loader, val_loader, optimizer, criterion, device)

