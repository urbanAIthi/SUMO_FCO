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
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm
from utils.create_cnn import create_cnn


def get_device():
    if torch.cuda.is_available():
        return ("cuda:0")
    else:
        return ("cpu")


class BigDataset(Dataset):
    def __init__(self, list_IDs, transforms=None):
        self.list_IDs = list_IDs
        self.transform = transforms

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        filename = self.list_IDs[index]
        with open(filename, "rb") as file:
            df = pickle.load(file)

        images, vectors, labels, names = self.prepare_data(df)
        return images, vectors, labels, names

    def prepare_data(self, df):
        # perform any necessary preprocessing
        print(len(df))
        df.dropna(axis=0, how='any', inplace=True)
        print(len(df))
        vectors = df['vector'].values
        detected = df['detected'].values
        images = df['image'].values

        vectors = torch.stack([torch.tensor(list(i)) for i in vectors], dim=0)
        detected = torch.stack([torch.tensor(i) for i in detected], dim=0)
        images = torch.stack([torch.tensor(i) for i in images])

        names = list(df.index.values)

        return images, vectors, detected, names


class Dataset(Dataset):

    def __init__(self, ID, transforms=None):
        with open(ID, "rb") as file:
            df = pickle.load(file)
        df.dropna(axis=0, how="any", inplace=True)
        self.transform = transforms
        vectors = df['vector'].values
        detected = df['detected'].values
        images = df['image'].values
        self.vectors = torch.stack([torch.tensor(list(i)) for i in vectors], dim=0)
        self.detected = torch.stack([torch.tensor(i) for i in detected], dim=0)
        self.images = torch.stack([i.clone().detach() for i in tqdm(images)])
        self.names = list(df.index.values)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transform(self.images[idx])
        vectors = self.vectors[idx]
        labels = self.detected[idx]
        names = self.names[idx]
        return images, vectors, labels, names


def train(model, IDs, val_loader, transforms):
    device = get_device()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 0.0001 for vit
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9, verbose=False)
    model.to(device)
    best_val = 0
    for e in range(50):
        model.train()
        running_loss = 0
        counter = 0
        acc = list()
        pre = list()
        rec = list()
        ones = list()
        for ID in IDs:
            print(ID)
            train_set = Dataset(ID, transforms)
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
            for images, vectors, labels, _ in iter(train_loader):
                print(len(images))
                counter += 1
                images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
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
                try:
                    loss.backward()
                except:
                    raise
                optimizer.step()
                scheduler.step()
        with torch.no_grad():
            epoch_eval_acc = list()
            epoch_eval_pre = list()
            epoch_eval_rec = list()
            epoch_eval_loss = list()
            running_eval_loss = 0
            for images, vectors, labels, _ in iter(val_loader):
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
        average_acc = sum(acc) / len(acc)
        average_pre = sum(pre) / len(pre)
        average_rec = sum(rec) / len(rec)
        average_ones = sum(ones) / len(ones)
        average_eval_acc = sum(epoch_eval_acc) / len(epoch_eval_acc)
        average_eval_pre = sum(epoch_eval_pre) / len(epoch_eval_pre)
        average_eval_rec = sum(epoch_eval_rec) / len(epoch_eval_rec)
        wandb.log({"train_loss": running_loss.item()})
        wandb.log({"train_acc": average_acc})
        wandb.log({"train_ones": average_ones})
        wandb.log({"eval_loss": running_eval_loss.item()})
        wandb.log({"eval_acc": average_eval_acc})
        if average_eval_acc.item() > best_val:
            print('saving best model')
            torch.save(model.state_dict(), 'best_valacc_model.pth')
            # wandb.save('best_valacc_model.pth')
        print(e, running_loss.item(), average_acc, average_rec, average_pre, average_ones, '---------- eval:',
              running_eval_loss.item(), average_eval_acc, average_eval_rec, average_eval_pre)
    torch.save(model.state_dict(), 'model_checkpoint.pth')  # Save the trained model
    wandb.save('model_checkpoint.pth')  # Save the trained model to cloud
    return model


def test(model, test_loader, analyse=False, analyse_path='test'):
    if analyse:
        if not os.path.exists(analyse_path):
            os.makedirs(analyse_path)
    device = get_device()
    model.to(device)
    model.eval()
    test_acc = list()
    test_pre = list()
    test_rec = list()
    output_label_comps = ('TP', 'FP', 'TN', 'FN')
    for olc in output_label_comps:
        if os.path.exists(os.path.join(analyse_path, olc)):
            shutil.rmtree(os.path.join(analyse_path, olc))
        os.mkdir(os.path.join(analyse_path, olc))
    with torch.no_grad():
        for images, vectors, labels, names in iter(test_loader):
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)

            # Perform the same preprocessing as in the training loop, if necessary

            outputs = model(images.float(), vectors.float())
            labels = labels.unsqueeze(1).float()

            accuracy, precision, recall, _ = binary_metrics(outputs, labels)
            if analyse:
                for i in range(len(images)):
                    output_label_comp = get_single_classification_metric(outputs[i], labels[i])
                    img = images[i].cpu().numpy()
                    img_pil = PIL.Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
                    # draw circle for ego vehicle
                    img_pil = draw_circle_at_position(img_pil, 5, 0, 0, (255, 0, 0))
                    # draw circle at vector position
                    img_pil = draw_circle_at_position(img_pil, 5, vectors[i][0], vectors[i][1], (0, 255, 0))
                    img_pil.save(os.path.join(analyse_path, output_label_comp, str(names[i]) + '.jpg'))
            test_acc.append(accuracy)
            test_pre.append(precision)
            test_rec.append(recall)

    average_test_acc = sum(test_acc) / len(test_acc)
    average_test_pre = sum(test_pre) / len(test_pre)
    average_test_rec = sum(test_rec) / len(test_rec)
    print('Test accuracy: ', average_test_acc, 'Test precision: ', average_test_pre, 'Test recall: ', average_test_rec)
    wandb.log({"test_acc": average_test_acc})
    wandb.log({"test_pre": average_test_pre})
    wandb.log({"test_rec": average_test_rec})
    shutil.make_archive('test_images', 'zip', 'test')
    wandb.save('test_images.zip')


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
    precision = true_positives / (
            true_positives + false_positives + 1e-9)  # Add small epsilon to avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-9)  # Add small epsilon to avoid division by zero

    return accuracy, precision, recall, count_ones


def get_single_classification_metric(output, label, threshold=0.5):
    binary_prediction = (output > threshold).float()

    if binary_prediction == 1 and label == 1:
        return "TP"
    elif binary_prediction == 1 and label == 0:
        return "FP"
    elif binary_prediction == 0 and label == 0:
        return "TN"
    elif binary_prediction == 0 and label == 1:
        return "FN"


def draw_circle_at_position(img, circle_radius, center_x=0, center_y=0, circle_color=(255, 0, 0), circle_width=1,
                            mtopixel=0.4):
    img_copy = img.copy()
    width, height = img.size

    # Transform the input coordinates to use the center of the image as the origin
    transformed_x = width // 2
    transformed_y = height // 2

    transformed_x = transformed_x + ((1 / mtopixel) * center_x)
    transformed_y = transformed_y - ((1 / mtopixel) * center_y)  # minus because of the different coordinate system

    draw = ImageDraw.Draw(img_copy)
    draw.ellipse((transformed_x - circle_radius, transformed_y - circle_radius, transformed_x + circle_radius,
                  transformed_y + circle_radius), outline=circle_color, width=circle_width)
    return img_copy


if __name__ == '__main__':
    m = 'resnet'
    transforms = transforms.Compose([
        transforms.CenterCrop(400),
    ])
    dataset = 'test'
    train_index = [0]
    val_index = [1]
    test_index = [2]
    train_IDs = [f'{dataset}/{dataset}_{i}.pkl' for i in train_index]
    val_ID = [f'{dataset}/{dataset}_{i}.pkl' for i in test_index]
    test_ID = [f'{dataset}/{dataset}_{i}.pkl' for i in test_index]
    val_set = Dataset(val_ID[0], transforms=transforms)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    if m == 'vit':
        model = ViT(image_size=400, patch_size=50, num_classes=1, channels=3, mlp_dim=2048, dropout=0.5, emb_dropout=0,
                    dim=512, depth=6, heads=8)
    elif m == 'resnet':
        model = create_cnn()
    else:
        raise ValueError('Model not implemented')
    wandb.init(project="sumo_detector",
               name=f"{m}_{datetime.datetime.fromtimestamp(int(time.time())).strftime('%m-%d-%H-%M')}",
               mode='offline')
    wandb.log({"dataset": dataset})
    trained_model = train(model, train_IDs, val_loader, transforms)
    model.load_state_dict(torch.load('best_valacc_model.pth'))
    wandb.save('best_valacc_model.pth')
    test_set = Dataset(test_ID[0], transforms=transforms)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    test(trained_model, test_loader, analyse=True)
