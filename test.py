import pdb
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import torch
import matplotlib.pyplot as plt


use_gpu = torch.cuda.is_available()
print("USING GPU? : ", use_gpu)

# Data Transformations
image_resize = (320,320)
crop_size = 224

train_file_path = '../CheXpert-v1.0-small/train.csv'
validation_file_path = '../CheXpert-v1.0-small/valid.csv'

pretrained_nn = False
num_classes = 14

# Training Settings
train_batch_size = 4
train_max_epoch = 3

# File Imports
from densenet import DenseNet121
from dataset import CheXpertDataset
from train import CheXpertTrainer
from dataset import dataLoaderTrain, dataLoaderVal, dataLoaderTest

MODEL_PATH = "/data/rsg/mammogram/jxiang/cheXpert/6.869_final_project/m-epoch2full_loss.pth.tar"

CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


if __name__ == "__main__":
    model = DenseNet121(num_classes).cuda()
    outGT, outPRED = CheXpertTrainer.test(model, dataLoaderTest, num_classes, MODEL_PATH, CLASS_NAMES)
    # the only class we are concerned with is Edema
    for i in range (num_classes):
        if CLASS_NAMES[i] != 'Edema':
            continue
        else:
            fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:,i], outPRED.cpu()[:,i])
            roc_auc = metrics.auc(fpr, tpr)
            outLABEL = outPRED.cpu()[:,i].data.numpy()
            outLABEL[outLABEL < 0.5] = 0
            outLABEL[outLABEL >= 0.5] = 1
            tn, fp, fn, tp = metrics.confusion_matrix(outGT.cpu()[:,i], outLABEL).ravel()
            sensitivity = tp / (tp + fn)
            print("SENSITIVITY: ", sensitivity)

            plt.title('ROC for: ' + CLASS_NAMES[i])
            plt.plot(fpr, tpr, label = 'U-ones: AUC = %0.2f' % roc_auc)
    
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

            plt.savefig("ROC_full_epoch_0.png", dpi=1000)
            plt.show()
