#necessary imports
import os
import timm
import numpy as np
import pandas as pd
import random as rd

from sklearn.preprocessing import OrdinalEncoder

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

np.random.seed(42)
torch.manual_seed(42)

#soft label encoder
def soft_label_encoder(df):
    df = df.drop(columns = features_remove)
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    non_numeric_columns = non_numeric_columns.drop('isic_id')

    category_encoder = OrdinalEncoder(
    categories='auto', # The encoder will automatically determine the categories for each feature.
    dtype=int, # ouput them as integers
    handle_unknown='use_encoded_value', # The encoder will use a specified integer value for these unknown categories.
    unknown_value=-2, # which is -2 for unknown values
    encoded_missing_value=-1, # and -1 for encoded missing value
    )

    X_cat = category_encoder.fit_transform(df[non_numeric_columns])
    for c, cat_col in enumerate(non_numeric_columns):
        df[cat_col] = X_cat[:, c]    
        
        
    df = df.replace([np.nan], -1) #replace nan values by -1 in rest of df
    df = df.replace([np.inf, -np.inf, 0], 0.01) #replace infinitesimal and 0 values by 0.01
    
    df['target'] = df['target'].replace(0.01,0)

    rd.seed(42)

    malignant_df = df[df['target']==1]
    benign_df = df[df['target']==0]
    total = range(len(benign_df))
    index = rd.sample(total,15_000 - len(malignant_df))
    benign_df_2 = benign_df.iloc[index].reset_index(drop=True)

    df = pd.concat([malignant_df,benign_df_2]).reset_index(drop=True)
        
    shuffled_df = df.sample(frac=1, random_state = 42).reset_index(drop=True) 
    
    return shuffled_df

#SLE FCN
class SLE_FCN(nn.Module):
    def __init__(self, SLE_out_size = 40, num_classes = 100):
        super(SLE_FCN, self).__init__()
        self.linear = nn.Linear(SLE_out_size,num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        
    def forward(self, SLE_output):
        output = self.linear(SLE_output)
        output = self.bn(output)
        return output

#ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, model_name, checkpoint_path, pretrained=False):
        super(ViTEncoder, self).__init__()
        self.model = timm.create_model(model_name, checkpoint_path =checkpoint_path, pretrained = pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False

        num_layers_to_unfreeze = 5
        
        params = list(self.model.parameters())[::-1]

        for param in params[:num_layers_to_unfreeze]:
            param.requires_grad = True
        
    def forward(self, images):
        output = self.model(images)
        return output

    
model = ViTEncoder('vit_base_patch16_224', checkpoint_path ='/kaggle/input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_224-80ecf9dd.pth', pretrained=True)
model.to(device)

#ViT FCN
class ViT_FCN(nn.Module):
    def __init__(self, ViT_out_size = 1000, num_classes = 100):
        super(ViT_FCN, self).__init__()
        self.linear = nn.Linear(ViT_out_size,num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        
    def forward(self, ViT_output):
        output = self.linear(ViT_output)
        output = self.bn(output)
        return output

#MHA Block
class MHA(nn.Module):
    def __init__(self, d_i=100, d_m=100, n_heads_i=5, n_heads_m=5):
        super(MHA, self).__init__()
        self.d_i = d_i
        self.d_m = d_m
        #trying out with different n_heads we'll see if this works
        self.n_heads_i = n_heads_i
        self.n_heads_m = n_heads_m
        
        #mappings for metadata
        #q_m needs to match k_i and vice-versa 
        self.q_mappings_m = nn.ModuleList([nn.Linear(d_m, d_i) for _ in range(self.n_heads_m)])
        self.k_mappings_m = nn.ModuleList([nn.Linear(d_m, d_m) for _ in range(self.n_heads_m)])
        self.v_mappings_m = nn.ModuleList([nn.Linear(d_m, d_m) for _ in range(self.n_heads_m)])
        #mappings for images
        self.q_mappings_i = nn.ModuleList([nn.Linear(d_i, d_m) for _ in range(self.n_heads_i)])
        self.k_mappings_i = nn.ModuleList([nn.Linear(d_i, d_i) for _ in range(self.n_heads_i)])
        self.v_mappings_i = nn.ModuleList([nn.Linear(d_i, d_i) for _ in range(self.n_heads_i)])
        self.softmax = nn.Softmax(dim=-1)
        self.linear_m = nn.Linear(self.n_heads_m * self.d_m, self.d_m)
        self.linear_i = nn.Linear(self.n_heads_i * self.d_i, self.d_i)

    def forward(self, image_input, metadata_input):
        result_i = []
        result_m = []
        
        for head in range(self.n_heads_i):
            q_mapping_m = self.q_mappings_m[head]
            k_mapping_i = self.k_mappings_i[head]
            v_mapping_i = self.v_mappings_i[head]
            
            k_i, v_i = k_mapping_i(image_input), v_mapping_i(image_input) 
            q_m = q_mapping_m(metadata_input)

            attention_i = self.softmax(q_m @ k_i.T / (self.d_i ** 0.5))
            result_i.append(attention_i @ v_i)
                
        for head in range(self.n_heads_m):
            q_mapping_i = self.q_mappings_i[head]
            k_mapping_m = self.k_mappings_m[head]
            v_mapping_m = self.v_mappings_m[head] 

            k_m, v_m = k_mapping_m(metadata_input), v_mapping_m(metadata_input)  
            q_i = q_mapping_i(image_input) 

            attention_m = self.softmax(q_i @ k_m.T / (self.d_m ** 0.5))
            result_m.append(attention_m @ v_m)
        
        # Concatenate results
        output_i = torch.cat(result_i, dim=-1)
        output_m = torch.cat(result_m, dim=-1)

        output_i = self.linear_i(output_i)
        output_m = self.linear_m(output_m)
        
        skip_step_i = output_i + image_input
        skip_step_m = output_m + metadata_input
        
        concat = torch.cat((skip_step_i, skip_step_m), dim=-1)

        return concat

#Final FCN
class Final_FCN(nn.Module):
    def __init__(self, MHA_out_size = 200, num_classes = 1):
        super(Final_FCN, self).__init__()
        self.linear1 = nn.Linear(MHA_out_size,MHA_out_size//2)
        self.linear2 = nn.Linear(MHA_out_size//2, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, MHA_output):
        L1 = self.linear1(MHA_output)
        L2 = self.linear2(L1)
        output = self.sigmoid(L2)
        return output

#Putting it all together
class Multimodal_Transformer(nn.Module):
    def __init__(self, n_blocks = 2, n_heads_i = 5, n_heads_m = 5):
        super(Multimodal_Transformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads_i = n_heads_i
        self.n_heads_m = n_heads_m
        self.mha = MHA(d_i=100, d_m=100, n_heads_i=5, n_heads_m=5)
        
        self.sle_fcn = SLE_FCN()
        
        self.vit_encoder = ViTEncoder('vit_base_patch16_224', checkpoint_path = '/kaggle/input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_224-80ecf9dd.pth', pretrained=True).to(device)
        self.vit_fcn = ViT_FCN()
        
        self.final_fcn = Final_FCN()
        
        
    def forward(self, images, metadata):
        sle_fcn_output = self.sle_fcn(metadata)
        
        vit_encoded = self.vit_encoder(images)
        vit_fcn_output = self.vit_fcn(vit_encoded)
                        
        MHA_output = self.mha(sle_fcn_output, vit_fcn_output)
        
        predictions = self.final_fcn(MHA_output)
        
        return predictions
