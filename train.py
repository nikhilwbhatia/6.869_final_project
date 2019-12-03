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
train_max_epoch = 3

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
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['state_dict'])

        # Training the network
        lossMIN = 100000

        for epoch in tqdm(range(0, train_max_epoch)):
            print('EPOCH: ', epoch)
            batchs, losst, losse = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, train_max_epoch, num_classes, loss)
            lossVal = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, train_max_epoch, num_classes, loss)

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, 'm-epoch'+str(epoch)+str(lossVal))
                print('Epoch [' + str(epoch + 1) + '] [SAVE] loss= ' + str(lossVal))

            else:
                print('Epoch [' + str(epoch + 1) + '] ------ loss= ' + str(lossVal))

        return batchs, losst, losse


    def epochTrain(model, dataLoader, optimizer, epochMax, classCount, loss):

        batch = []
        losstrain = []
        losseval = []

        model.train()

        for batchID, (varInput, target) in enumerate(dataLoaderTrain):

            varTarget = target.cuda(non_blocking=True)

            # varTarget = target.cuda()


            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

            l = lossvalue.item()
            losstrain.append(l)

            if batchID % 35 == 0:
                print(batchID // 35, "% batches computed")
                # Fill three arrays to see the evolution of the loss


                batch.append(batchID)

                le = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, train_max_epoch, num_classes, loss).item()
                losseval.append(le)

                print(batchID)
                print(l)
                print(le)

        return batch, losstrain, losseval


    def epochVal(model, dataLoader, optimizer, epochMax, classCount, loss):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                target = target.cuda(non_blocking=True)
                varOutput = model(varInput)

                losstensor = loss(varOutput, target)
                lossVal += losstensor
                lossValNorm += 1

        outLoss = lossVal / lossValNorm
        return outLoss

    # ---- Computes area under ROC curve
    # ---- dataGT - ground truth data
    # ---- dataPRED - predicted data
    # ---- classCount - number of classes

    def computeAUROC(dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC

    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):

        cudnn.benchmark = True

        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)

                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        print('AUROC mean ', aurocMean)

        for i in range(0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])

        return outGT, outPRED