import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time
import datetime
from torchvision import transforms
import PIL.Image
from vit_pytorch import ViT
from utils.create_cnn import CustomResNet
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm
from configs.config_networks import NETWORK_CONFIG, VIT_CONFIG, RESNET_CONFIG
from typing import List
from test_model import test_model, analyze_test_results

import logging
import importlib
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(e):
    if e < 5: # Warm-up phase
        return e / 5
    else: # Learning rate reduction phase
        return 0.9**(e - 5) # exponential decay after warm-up


class Dataset(Dataset):

    def __init__(self, ID: str, transforms = None):
        with open(ID, "rb") as file:
            df = pickle.load(file)
        df.dropna(axis=0, how="any", inplace = True)
        # use only the first items in the df for debugging
        # df = df.iloc[:50]
        self.transform = transforms
        print(df.keys())
        vectors = df['vector'].values
        detected = df['detected'].values
        images = df['plot'].values
        if 'proportion' in df.keys():
            proportions = df['proportion']
            self.proportions = torch.stack([torch.tensor(list(i)) for i in proportions], dim=0)
        else:
            self.proportions = torch.stack([torch.tensor(0) for i in vectors], dim=0)
        self.vectors = torch.stack([torch.tensor(list(i)) for i in vectors], dim=0)
        self.detected = torch.stack([torch.tensor(i) for i in detected], dim=0)
        self.images = torch.stack([i.clone().detach() for i in tqdm(images)])
        self.names = list(df.index.values)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform is not None:
            images = self.transform(self.images[idx])
        else:
            images = self.images[idx]
        vectors = self.vectors[idx]
        labels = self.detected[idx]
        names = self.names[idx]
        proportions = self.proportions[idx]
        return images, vectors, labels, names, proportions

def train(model: torch.nn.Module, train_IDs: List[str], val_IDs: List[str], filename: str, transforms: bool = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=NETWORK_CONFIG['INIT_LR'])
    scheduler = StepLR(optimizer, step_size=NETWORK_CONFIG['LR_STEP_SIZE'], gamma=NETWORK_CONFIG['LR_GAMMA'], verbose=False)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.to(device)
    best_val_loss = np.inf
    for e in range(NETWORK_CONFIG['NUM_EPOCHS']):
        model.train()
        running_loss = 0
        counter = 0
        acc, pre, rec, ones = list(), list(), list(), list()
        for ID in train_IDs:
            print(ID)
            train_set = Dataset(ID, transforms)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
            for images, vectors, labels, _, _ in iter(train_loader):
                counter += 1
                images, vectors, labels= images.to(device), vectors.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(images.float(), vectors.float())
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                accuracy, precision, recall, o = binary_metrics(outputs, labels)
                acc.append(accuracy)
                pre.append(precision)
                rec.append(recall)
                ones.append(o)
                running_loss += loss
                loss.backward()
                optimizer.step()
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        with torch.no_grad():
            epoch_eval_acc = list()
            epoch_eval_pre = list()
            epoch_eval_rec = list()
            epoch_eval_loss = list()
            running_eval_loss = 0
            for val_ID in val_IDs:
                val_set = Dataset(val_ID, transforms)
                val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
                for images, vectors, labels, _, _ in iter(val_loader):
                    images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
                    outputs = model.forward(images.float(), vectors.float())
                    labels = labels.unsqueeze(1).float()
                    loss = criterion(outputs, labels)
                    running_eval_loss += loss
                    accuracy, precision, recall, _ = binary_metrics(outputs, labels)
                    epoch_eval_acc.append(accuracy)
                    epoch_eval_pre.append(precision)
                    epoch_eval_rec.append(recall)
                    epoch_eval_loss.append(loss)
            if running_eval_loss < best_val_loss:
                best_val_loss = running_eval_loss
                torch.save(model.state_dict(), os.path.join('models', filename, 'best_val_model_state_dict.pth'))
                torch.save(model, os.path.join('models', filename, 'best_val_model_model.pt'))
        average_acc = sum(acc)/len(acc)
        average_pre = sum(pre)/len(pre)
        average_rec = sum(rec)/len(rec)
        average_ones = sum(ones)/len(ones)
        average_eval_acc = sum(epoch_eval_acc)/len(epoch_eval_acc)
        average_eval_pre = sum(epoch_eval_pre)/len(epoch_eval_pre)
        average_eval_rec = sum(epoch_eval_rec)/len(epoch_eval_rec)
        wandb.log({"train_loss": running_loss.item()})
        wandb.log({"train_acc": average_acc})
        wandb.log({"train_ones": average_ones})
        wandb.log({"eval_loss": running_eval_loss.item()})
        wandb.log({"eval_acc": average_eval_acc})
        print(e, running_loss.item(), average_acc,average_rec, average_pre, average_ones, '---------- eval:', running_eval_loss.item(), average_eval_acc, average_eval_rec, average_eval_pre)
    return model

def binary_metrics(predictions, targets, threshold=0.5):
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).float()
    count_ones = torch.sum(binary_preds == 1).item()

    # Calculate true positives, false positives, and false negatives
    true_positives = (binary_preds * targets).sum()
    false_positives = (binary_preds * (1 - targets)).sum()
    false_negatives = ((1 - binary_preds) * targets).sum()

    # Calculate accuracy, precision, and recall
    accuracy = (true_positives + (1 - binary_preds).sum() - false_negatives) / len(targets)
    precision = true_positives / (true_positives + false_positives + 1e-9)  # Add small epsilon to avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-9)  # Add small epsilon to avoid division by zero

    return accuracy, precision, recall, count_ones



if __name__ == '__main__':
    # create folder for the training process
    filename = f'{NETWORK_CONFIG["NETWORK_TYPE"]}_{NETWORK_CONFIG["DATASET"]}_{NETWORK_CONFIG["FILE_EXTENSION"]}'
    if not os.path.exists(os.path.join('models', filename)):
        os.makedirs(os.path.join('models', filename))
    else:
        raise ValueError('The filename already exists')
    # copy the configs folder from the dataset to the training folder
    os.mkdir(os.path.join('models', filename, 'configs'))
    shutil.copytree(os.path.join('data', NETWORK_CONFIG['DATASET'], 'configs'), os.path.join('models', filename, 'configs'), dirs_exist_ok=True)
    # import the dataset config    
    DATASET_GENERAL = importlib.import_module(f"models.{filename}.configs.config_dataset").DATASET_GENERAL
    # copy the network training config to the configs folder
    shutil.copy(os.path.join('configs', 'config_networks.py'), os.path.join('models', filename, 'configs'))
    # start the logger
    logging.basicConfig(filename=os.path.join('models', filename, 'log.log'), level=logging.INFO)
    logging.info('starting simulation at ' + time.strftime("%H:%M:%S", time.localtime()))

    # copy the current configs
    train_IDs = [f"data/{NETWORK_CONFIG['DATASET']}/{NETWORK_CONFIG['DATASET']}_{i}.pkl" for i in NETWORK_CONFIG['TRAIN_IDs']]
    val_IDs = [f"data/{NETWORK_CONFIG['DATASET']}/{NETWORK_CONFIG['DATASET']}_{i}.pkl" for i in NETWORK_CONFIG['VAL_IDs']]
    test_IDs = [f"data/{NETWORK_CONFIG['DATASET']}/{NETWORK_CONFIG['DATASET']}_{i}.pkl" for i in NETWORK_CONFIG['TEST_IDs']]
    if NETWORK_CONFIG['NETWORK_TYPE'] == 'ViT':
        model = ViT(**VIT_CONFIG, image_size=DATASET_GENERAL['BEV_IMAGE_SIZE'])
    elif NETWORK_CONFIG['NETWORK_TYPE'] == 'ResNet':
        model = CustomResNet()
    else:
        raise ValueError('Model type not supported')
    wandb.init(project="sumo_detector", name=f"{datetime.datetime.fromtimestamp(int(time.time())).strftime('%m-%d-%H-%M')}", mode='online')
    trained_model = train(model, train_IDs, val_IDs, filename)
    torch.save(trained_model, os.path.join('models', filename, 'model.pt'))
    torch.save(trained_model.state_dict(), os.path.join('models', filename, 'model_state_dict.pt'))
    wandb.save(os.path.join('models', filename, 'model.pt'))
    wandb.save(os.path.join('models', filename, 'model_state_dict.pt'))
    logging.info(f'finished training at {time.strftime("%H:%M:%S", time.localtime())}')
    average_test_acc, average_test_pre, average_test_rec = test_model(filename)
    wandb.log({"test_acc": average_test_acc})
    wandb.log({"test_pre": average_test_pre})
    wandb.log({"test_rec": average_test_rec})
    analyze_test_results(filename)
