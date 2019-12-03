import pdb
import numpy as np


import torch


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

# File Imports

from densenet import DenseNet121
from dataset import CheXpertDataset
from train import CheXpertTrainer

from dataset import dataLoaderTrain, dataLoaderVal, dataLoaderTest



if __name__ == "__main__":
    model = DenseNet121(num_classes).cuda()
    batch, losst, losse = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, num_classes, train_max_epoch, checkpoint=None)
    print('Model Trained')

    losstn = []
    for i in range(0, len(losst), 35):
        losstn.append(np.mean(losst[i:i + 35]))

    print(losstn)
    print(losse)




