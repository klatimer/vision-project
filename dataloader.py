import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack

import os
import pandas as pd
from PIL import Image
import numpy as np


class BirdLoader(object):

    def __init__(self, args):
        super(BirdLoader, self).__init__()
        transform = transforms.Compose(
            [
                # Data augmentations
                transforms.Resize((64, 64)),
                # transforms.RandomCrop((100, 100)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        root_dir = '~/.kaggle/competitions/birds/'
        csv_file = 'labels.csv'
        test_csv = 'sample.csv'
        test_dir = '~/.kaggle/competitions/birds/test/'

        bird_train_set = BirdTrainSet(csv_file=csv_file, root_dir=root_dir, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(bird_train_set, batch_size=args.batchSize,
                                                       shuffle=True, num_workers=2)
        bird_test_set = BirdTestSet(csv_file=test_csv, root_dir=root_dir, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(bird_test_set, batch_size=args.batchSize,
                                                      shuffle=False, num_workers=2)

        classes = []
        with open(os.path.expanduser(os.path.join(root_dir, 'names.txt'))) as f:
            for line in f:
                classes.append(line.rstrip())
        self.classes = classes


# Need to wrap images and labels
class BirdTrainSet(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        train_labels = pd.read_csv(root_dir + csv_file)
        self.names = train_labels.iloc[:, 0]  # image file names
        self.labels = train_labels.iloc[:, 1]  # corresponding labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = os.path.join(self.root_dir, self.names.iloc[idx])
        img = (Image.open(os.path.expanduser(name)))
        img = self.transform(img)
        label = self.labels.iloc[idx]
        item = {'image': img, 'label': torch.from_numpy(np.array(label))}
        # item = {img, torch.from_numpy(np.array(label))}
        return item


class BirdTestSet(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        train_labels = pd.read_csv(root_dir + csv_file)
        self.names = train_labels.iloc[:, 0]  # image file names
        self.labels = train_labels.iloc[:, 1]  # corresponding labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = os.path.join(self.root_dir, self.names.iloc[idx])
        img = (Image.open(os.path.expanduser(name)))
        img = self.transform(img)
        label = self.labels.iloc[idx]
        item = {'image': img, 'label': torch.from_numpy(np.array(label))}
        # item = {img, torch.from_numpy(np.array(label))}
        return item