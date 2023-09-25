import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
import importlib
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from vit_pytorch import ViT
from utils.create_cnn import CustomResNet

import wandb
import pickle
import matplotlib.pyplot as plt
import random

def test_model(model_name: str):
    # create new test folder delete
    test_folder = os.path.join("models",model_name, 'test')
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    # create the test folder
    os.mkdir(test_folder)
    print(f"Testing model {model_name} and created folder")
    # import the configs of the model
    NETWORK_CONFIG = importlib.import_module(f"models.{model_name}.configs.config_networks").NETWORK_CONFIG
    VIT_CONFIG = importlib.import_module(f"models.{model_name}.configs.config_networks").VIT_CONFIG
    RESNET_CONFIG = importlib.import_module(f"models.{model_name}.configs.config_networks").RESNET_CONFIG
    DATASET_GENERAL = importlib.import_module(f"models.{model_name}.configs.config_dataset").DATASET_GENERAL
    DEFAULT_CV = importlib.import_module(f"models.{model_name}.configs.config_cv").DEFAULT
    test_IDs = [f"data/{NETWORK_CONFIG['DATASET']}/{NETWORK_CONFIG['DATASET']}_{i}.pkl" for i in NETWORK_CONFIG['TEST_IDs']]
    if NETWORK_CONFIG['NETWORK_TYPE'] == 'ViT':
        model = ViT(**VIT_CONFIG, image_size=DATASET_GENERAL['BEV_IMAGE_SIZE'])
        model.load_state_dict(torch.load(f"models/{model_name}/model_state_dict.pt"))
    elif NETWORK_CONFIG['NETWORK_TYPE'] == 'ResNet':
        model = CustomResNet()
        model.load_state_dict(torch.load(f"models/{model_name}/model_state_dict.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_dict = {}
    # load the data
    acc, pre, rec, ones = list(), list(), list(), list()
    print('starting testing')
    for ID in test_IDs:
        test_set = Dataset(ID)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        for images, vectors, labels, names in iter(test_loader):
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model.forward(images.float(), vectors.float())
                labels = labels.unsqueeze(1).float()
            accuracy, precision, recall, _ = binary_metrics(outputs, labels)
            acc.append(accuracy)
            pre.append(precision)
            rec.append(recall)
            for image, vector, label, name, output in zip(images, vectors, labels, names, outputs):
                test_dict[name] = {}
                # convert the image to a pil image
                image = image.cpu().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = image.astype(np.uint8)
                image = image.squeeze()
                image = Image.fromarray(image, 'L')
                # get the m/pixel scale
                mp_ratio = (2*DEFAULT_CV['detection_range'])/DATASET_GENERAL['BEV_IMAGE_SIZE']
                # scale the vector to the image size
                scaled_vector = vector.cpu().numpy() / mp_ratio
                # draw arrow from center of the image to the scaled vector
                x_center, y_center = image.size[0]/2, image.size[1]/2
                end_x, end_y = (x_center + scaled_vector[0]), (y_center + -scaled_vector[1]) # negative because of the different coordinate system in PIL
                draw = ImageDraw.Draw(image)
                draw.line((x_center, y_center, end_x, end_y), fill="red", width=3)
                test_dict[name]['image'] = image
                test_dict[name]['vector'] = vector.cpu().numpy()
                # get the len of the vector vector
                test_dict[name]['vector_len'] = np.linalg.norm(test_dict[name]['vector'])
                test_dict[name]['label'] = label.cpu().numpy()
                test_dict[name]['output'] = output.cpu().numpy()
                test_dict[name]['result'] = get_single_classification_metric(output, label)
    average_test_acc = sum(acc) / len(acc)
    average_test_pre = sum(pre) / len(pre)
    average_test_rec = sum(rec) / len(rec)

    """
    # log the results to wandb
    wandb.init(project="table-test")
    columns=list(list(test_dict.values())[0].keys())
    data=[list(inner_dict.values()) for inner_dict in test_dict.values()]
    # conver the images to be wandb compatible
    for d in data:
        for d_i in d:
            if isinstance(d_i, Image.Image):
                d[d.index(d_i)] = wandb.Image(d_i)
    my_table = wandb.Table(
    columns=columns,
    data=data,
    )
    wandb.log({"Table Name": my_table})"""
    # save the test_dict to a pickle file
    with open(f"models/{model_name}/test/test_dict.pkl", "wb") as f:
        pickle.dump(test_dict, f)
    # save the average results to a txt file
    with open(f"models/{model_name}/test/average_results.txt", "w") as f:
        f.write(f"Average Accuracy: {average_test_acc}\n")
        f.write(f"Average Precision: {average_test_pre}\n")
        f.write(f"Average Recall: {average_test_rec}\n")
    return average_test_acc, average_test_pre, average_test_rec


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

def draw_circle_at_position(img, circle_radius, center_x=0, center_y=0, circle_color=(255, 0, 0), circle_width=1, mtopixel = 0.4):
    img_copy = img.copy()
    width, height = img.size

    # Transform the input coordinates to use the center of the image as the origin
    transformed_x = width // 2
    transformed_y = height // 2

    transformed_x = transformed_x + ((1/mtopixel) * center_x)
    transformed_y = transformed_y - ((1/mtopixel) * center_y) #minus because of the different coordinate system

    draw = ImageDraw.Draw(img_copy)
    draw.ellipse((transformed_x - circle_radius, transformed_y - circle_radius, transformed_x + circle_radius, transformed_y + circle_radius), outline=circle_color, width=circle_width)
    return img_copy

def analyze_test_results(model_path: str, num_test_images: int=10):
    '''
    This function uses the already created test_dict and creates a histogram for the output values
    and a png file for a specified number of items in the dict. The png file contains the image
    also visualizes the vector and shows the other values of the test_dict item. 
    '''
    # check if the test_dict exists
    if not os.path.exists(f"models/{model_path}/test/test_dict.pkl"):
        raise Exception("The test_dict.pkl file does not exist. Please run the test_model function first.")
    # load the test_dict
    with open(f"models/{model_path}/test/test_dict.pkl", "rb") as f:
        test_dict = pickle.load(f)
    # iterate over the test_dict and show the results
    i = 0
    # create histograms for the output label and devide the test_dict into TP, FP, TN, FN
    # Splitting the output values based on the results
    TP_values = np.concatenate([entry['output'] for entry in test_dict.values() if entry['result'] == 'TP'])
    TN_values = np.concatenate([entry['output'] for entry in test_dict.values() if entry['result'] == 'TN'])
    FN_values = np.concatenate([entry['output'] for entry in test_dict.values() if entry['result'] == 'FN'])
    FP_values = np.concatenate([entry['output'] for entry in test_dict.values() if entry['result'] == 'FP'])

    # Creating the subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plotting the histograms
    axs[0, 0].hist(TP_values, bins=20, color='blue')
    axs[0, 0].set_title('True Positive (TP)')

    axs[0, 1].hist(TN_values, bins=20, color='green')
    axs[0, 1].set_title('True Negative (TN)')

    axs[1, 0].hist(FN_values, bins=20, color='red')
    axs[1, 0].set_title('False Negative (FN)')

    axs[1, 1].hist(FP_values, bins=20, color='orange')
    axs[1, 1].set_title('False Positive (FP)')

    # Labeling axes
    for ax in axs.flat:
        ax.set(xlabel='Output', ylabel='Frequency')

    # Making space between subplots
    plt.tight_layout()
    plt.savefig(f"models/{model_path}/test/histograms.png")
    #randomly choose n test_dict items
    sample_test_items = dict(random.sample(list(test_dict.items()), num_test_images))
           
    for name, values in sample_test_items.items():
        img = values['image']
        values_str = "\n".join([f"{k}: {v}" for k, v in values.items() if k != 'image'])
        # Create a subplot for the image
        plt.subplot(2, 1, 1)
        plt.subplots_adjust(hspace=1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Values for {name}")
        plt.axis('off')

        # Create a subplot for the text
        plt.subplot(2, 1, 2)
        plt.text(0, 0.5, values_str, fontsize=10)
        plt.axis('off')
        plt.savefig(f"models/{model_path}/test/{name}.png", dpi=300)
        plt.close()






if __name__ == '__main__':
   from train_detection import Dataset, binary_metrics
   average_test_acc, average_test_pre, average_test_rec = test_model('ViT_ing_plot_vehicles_box_i3040_120000_newvector_2')
   analyze_test_results('ViT_ing_plot_vehicles_box_i3040_120000_newvector_2')
