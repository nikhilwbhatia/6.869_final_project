import pdb
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

image_resize = (320,320)
crop_size = 224

train_file_path = '../CheXpert-v1.0-small/train.csv'
validation_file_path = '../CheXpert-v1.0-small/valid.csv'

pretrained_nn = False
num_classes = 14

# Training Settings
train_batch_size = 4
train_max_epoch = 3

# Data Transformations
image_resize = (320,320)
crop_size = 224

# File Imports
from densenet import DenseNet121
from dataset import CheXpertDataset
from train import CheXpertTrainer
from test import CLASS_NAMES
from dataset import dataLoaderTrain, dataLoaderVal, dataLoaderTest


class HeatmapGenerator():
    def __init__(self, pathModel, num_classes, transCrop):
        model = DenseNet121(num_classes).cuda()
       
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.model.eval()

        self.weights = list(self.model._modules['densenet121'].features.parameters())[-2]

        #---- Initialize the image transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)  
        self.transformSequence = transforms.Compose(transformList)

    def generate(self, pathImageFile, pathOutputFile, transCrop):
        with torch.no_grad():

            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if use_gpu:
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = self.model._modules['densenet121'].features(imageData)
            label = 'Edema' if float(l[0][5]) >= 0.5 else 'No Edema'
            # heat map generation
            for i in range(0, len(self.weights)):
                map = output[0,i,:,:]
                if i == 0: heatmap = self.weights[i] * map
                else: heatmap += self.weights[i] * map
                npHeatmap =  heatmap.cpu().data.numpy()

        # blend original photo with heat map

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(pathOutputFile)
        plt.show()

if __name__ == "__main__":
    pathInputImage = "/data/rsg/mammogram/jxiang/cheXpert/CheXpert-v1.0-small/train/patient00209/study4/view1_frontal.jpg"
    pathOutputImage = "/data/rsg/mammogram/jxiang/cheXpert/heatmaptest_3.png"
    pathModel = "/data/rsg/mammogram/jxiang/cheXpert/6.869_final_project/m-epoch2full_loss.pth.tar"
    h = HeatmapGenerator(pathModel, num_classes, crop_size)
    h.generate(pathInputImage, pathOutputImage, crop_size) 
