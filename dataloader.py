import torch
import torchvision.transforms as transforms

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
                transforms.Resize((72, 72)),
                transforms.RandomCrop((64, 64)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(),
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

        dataSet = pd.read_csv(root_dir + csv_file)
        train_df = dataSet.sample(frac=0.8)
        test_df = dataSet.drop(train_df.index)

        bird_train_set = BirdTrainSet(train_labels=train_df, root_dir=root_dir, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(bird_train_set, batch_size=args.batchSize,
                                                       shuffle=True, num_workers=2)
        bird_test_set = BirdTestSet(train_labels=test_df, root_dir=root_dir, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(bird_test_set, batch_size=args.batchSize,
                                                      shuffle=False, num_workers=2)

        classes = []
        with open(os.path.expanduser(os.path.join(root_dir, 'names.txt'))) as f:
            for line in f:
                classes.append(line.rstrip())
        self.classes = classes


class BirdDemoLoader(object):
    """ Data loader for the demo """

    def __init__(self, args):
        super(BirdDemoLoader, self).__init__()

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        root_dir = '~/.kaggle/competitions/birds/'
        csv_file = 'labels.csv'

        bird_demo_set = BirdDemoSet(csv_file=csv_file, root_dir=root_dir, transform=transform)
        self.demoLoader = torch.utils.data.DataLoader(bird_demo_set, batch_size=1,
                                                      shuffle=False, num_workers=2)

        classes = []
        with open(os.path.expanduser(os.path.join(root_dir, 'names.txt'))) as f:
            for line in f:
                classes.append(line.rstrip())
        self.classes = classes

# Need to wrap images and labels
class BirdTrainSet(torch.utils.data.Dataset):

    def __init__(self, train_labels, root_dir, transform):
        self.root_dir = root_dir

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

    def __init__(self, train_labels, root_dir, transform):
        self.root_dir = root_dir

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

class BirdDemoSet(torch.utils.data.Dataset):
    """ Demo data set"""

    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        data = pd.read_csv(root_dir + csv_file)
        self.names = data.iloc[:, 0]  # image file names
        self.labels = data.iloc[:, 1]  # corresponding labels
        self.transform = transform
        self.preserve = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """ return a dictionary of the transformed img, label, and the original img"""
        name = os.path.join(self.root_dir, self.names.iloc[idx])
        img = (Image.open(os.path.expanduser(name)))
        ogImg = self.preserve(img)
        img = self.transform(img)
        label = self.labels.iloc[idx]
        item = {'image': img, 'label': torch.from_numpy(np.array(label)),
                'ogImg': ogImg}
        return item
