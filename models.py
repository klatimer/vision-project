import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

import math


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        """
        # Uncomment this to decrease learning rate by 10% every 50 epochs, following
        # an exponential curve.
        if epoch > 0:
            lr = args.lr * math.exp(math.log(0.1) * 50 / epoch)
        else:
            lr = args.lr
        """
        # This one uses the specified learning rate for all of the training
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# The highly experimental BirdNestV1 neural network
class BirdNestV1(BaseModel):
    def __init__(self, num_classes=555):
        super(BirdNestV1, self).__init__()
        self.features = nn.Sequential(
            # nn.AdaptiveAvgPool2d((227, 227)), # downsample
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(432, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
