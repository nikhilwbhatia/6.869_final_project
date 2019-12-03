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
from tqdm import tqdm

from dataset import dataLoaderTrain, dataLoaderVal, dataLoaderTest

use_gpu = torch.cuda.is_available()

num_classes = 14
train_max_epoch = 14

class CheXpertTrainer():

    def train(model, dataLoaderTrain, dataLoaderVal, num_classes, train_max_epoch, checkpoint):

        # Settings: Optimizer and Scheduler
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5)

        # Settings: Loss function (We will be adjusting primarily for this in our experiments)
        loss = torch.nn.BCELoss(size_average = True)
        # Can change in the future by passing in flags

        # Load checkpoint (functionality to resume an experiment from a checkpoint if we want)
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
        return
