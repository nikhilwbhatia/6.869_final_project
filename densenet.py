import os
import torch
import torch.nn as nn
import torchvision
import re

PATTERN = re.compile(
                         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth', "/data/rsg/mammogram/jxiang/cheXpert/models")
        for key in list(state_dict.keys()):
            res = PATTERN.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        self.densenet121.load_state_dict(state_dict)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
