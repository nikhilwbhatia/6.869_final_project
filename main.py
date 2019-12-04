import pdb
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import torch

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

CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


if __name__ == "__main__":
    model = DenseNet121(num_classes).cuda()
    batch, losst, losse = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, num_classes, train_max_epoch, checkpoint=None)
    print('Model Trained')

    losstn = []
    for i in range(0, len(losst), 35):
        losstn.append(np.mean(losst[i:i + 35]))

    print(losstn)
    print(losse)
    
    lt = losstn[0] + losstn[2] + losstn[3]
    le = losse[0] + losse[2] + losse[3] 
    batch = [i*35 for i in range(len(lt))]

    plt.plot(batch, lt, label = "train")
    plt.plot(batch, le, label = "eval")
    plt.xlabel("Nb of batches (size_batch = 64)")
    plt.ylabel("BCE loss")
    plt.title("BCE loss evolution")
    plt.legend()

    plt.savefig("chart5.png", dpi=1000)
    plt.show()




