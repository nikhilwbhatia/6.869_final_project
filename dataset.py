import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random

use_gpu = torch.cuda.is_available()

# Data Transformations
image_resize = (320,320)
crop_size = 224

train_file_path = '../CheXpert-v1.0-small/train.csv'
validation_file_path = '../CheXpert-v1.0-small/valid.csv'

pretrained_nn = False
num_classes = 14

# Training Settings
train_batch_size = 64
train_max_epoch = 3

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

class CheXpertDataset(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        """
        image_list_file: path to file
        policy: policy we are using to handle uncertain cases
        """
        images = []
        labels = []

        with open(image_list_file, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)
            i = 0
            for  line in csv_reader:
                i += 1
                image_name = line[0]
                # label in this case is equal to classifications for multiple features
                label = line[5:]

                for j in range(14):
                    if label[j]:
                        class_label = float(label[j])
                        if class_label == -1:
                            assert policy == "ones" or policy == "zeroes"
                            label[j] = 1 if policy == "ones" else 0
                        else:
                            label[j] = int(class_label)
                    else:
                        label[j] = 0

                images.append('../' + image_name)
                labels.append(label)

            self.images = images
            self.labels = labels
            self.transform = transform

    def __getitem__(self, i):
        image_name = self.images[i]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[i]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.cuda.FloatTensor(label)

    def __len__(self):
        return len(self.images)




                        

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformations = []
transformations.append(transforms.Resize(image_resize))
transformations.append(transforms.RandomResizedCrop(crop_size))
transformations.append(transforms.RandomHorizontalFlip())
transformations.append(transforms.ToTensor())
transformations.append(normalize)
transform_sequence = transforms.Compose(transformations)

dataset = CheXpertDataset(train_file_path, transform_sequence, policy="ones")
test_dataset, train_dataset = random_split(dataset, [500, len(dataset) - 500])
validation_dataset = CheXpertDataset(validation_file_path, transform_sequence)

dataLoaderTrain = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=validation_dataset, batch_size=train_batch_size, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=test_dataset, num_workers=24, pin_memory=True)




