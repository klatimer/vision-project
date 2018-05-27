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
        # return nn.MSELoss()

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
    def __init__(self):
        super(BirdNestV1, self).__init__()
        res = 128 # resolution (on a side) to downsample to - data set has nonuniform resolution
        filt_size = 5
        pool_size = 2
        self.downsample = nn.AdaptiveAvgPool2d((res, res))
        self.conv1 = nn.Conv2d(3, 48, filt_size) # 3 channel input, 6 layer feature map, filter size 5
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv2 = nn.Conv2d(6, 128, filt_size)
        self.fc1 = nn.Linear(128 * filt_size * filt_size, 960)
        self.fc2 = nn.Linear(960, 800)
        self.fc3 = nn.Linear(800, 555)

    def forward(self, x):
        x = self.downsample(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x