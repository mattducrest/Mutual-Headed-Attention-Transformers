#necessary imports
import os
import numpy as np
import random as rd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, recall_score

import torch

np.random.seed(42)
torch.manual_seed(42)

def train_one_epoch(epoch, multimodal_transformer, train_loader, optimizer, criterion, device, num_steps, pos_weight=5):
    multimodal_transformer.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for images, metadata, targets in pbar:
        images, metadata, targets = preprocess_data(images, metadata, targets, device)
        
        optimizer.zero_grad()
        outputs = multimodal_transformer(images, metadata)
        loss = compute_loss(outputs, targets, criterion, pos_weight)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        collect_predictions(all_predictions, all_targets, outputs, targets, device)
        
        current_accuracy = accuracy_score(all_targets, all_predictions)
        current_recall = recall_score(all_targets, all_predictions, zero_division=1)
        
        num_steps += 1
        pbar.set_postfix(loss=running_loss / num_steps, accuracy=current_accuracy, recall=current_recall)

    return running_loss, current_accuracy, current_recall

def validate_one_epoch(epoch, multimodal_transformer, val_loader, criterion, device, num_steps, pos_weight=5):
    multimodal_transformer.eval()
    val_loss = 0.0
    val_targets = []
    val_predictions = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')
        for images, metadata, targets in pbar:
            images, metadata, targets = preprocess_data(images, metadata, targets, device)
            outputs = multimodal_transformer(images, metadata)
            loss = compute_loss(outputs, targets, criterion, pos_weight)
            
            val_loss += loss.item() * images.size(0)

            collect_predictions(val_predictions, val_targets, outputs, targets, device)

            current_val_accuracy = accuracy_score(val_targets, val_predictions)
            current_val_recall = recall_score(val_targets, val_predictions, zero_division=1)

            num_steps += 1
            pbar.set_postfix(loss=val_loss / num_steps, accuracy=current_val_accuracy, recall=current_val_recall)

    return val_loss, current_val_accuracy, current_val_recall

def preprocess_data(images, metadata, targets, device):
    images = images.to(device)
    metadata = metadata.to(device).float()
    targets = targets.to(device).float().view(-1, 1)
    return images, metadata, targets

def compute_loss(outputs, targets, criterion, pos_weight):
    weights = (targets * pos_weight) + (1 - targets)
    loss_no_weight = criterion(outputs, targets)
    loss = (weights * loss_no_weight).mean()
    return loss

def collect_predictions(predictions_list, targets_list, outputs, targets, device):
    predictions = (outputs > 0.5).float()
    if device == 'cuda':
        predictions_list.extend(predictions.cpu().detach().numpy())
        targets_list.extend(targets.cpu().detach().numpy())
    else:
        predictions_list.extend(predictions.detach().numpy())
        targets_list.extend(targets.detach().numpy())

def train_model(num_epochs, multimodal_transformer, train_loader, val_loader, optimizer, criterion, device):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_recall": [],
        "val_recall": []
    }
    for epoch in range(num_epochs):
        running_loss, train_accuracy, train_recall = train_one_epoch(
            epoch, multimodal_transformer, train_loader, optimizer, criterion, device, num_steps=0
        )
        val_loss, val_accuracy, val_recall = validate_one_epoch(
            epoch, multimodal_transformer, val_loader, criterion, device, num_steps=0
        )
        
        # Store the metrics in the history dictionary
        history["train_loss"].append(running_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss / len(val_loader.dataset))
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss / len(train_loader.dataset):.2f}, '
              f'Val Loss: {val_loss / len(val_loader.dataset):.2f}, Val Accuracy: {val_accuracy:.2f}, '
              f'Val Recall: {val_recall:.2f}')
    return history

