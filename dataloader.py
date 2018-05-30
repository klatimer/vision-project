import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack

import os
import pandas as pd
from PIL import Image

class BirdLoader(object):

    def __init__(self, args):
        super(BirdLoader, self).__init__()
        transform = transforms.Compose(
        [
            # Data augmentations
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        root_dir = '~/.kaggle/competitions/birds/'
        csv_file = 'labels.csv'
        test_dir = 'test'

        bird_train_set = BirdTrainSet(csv_file=csv_file, root_dir=root_dir, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(bird_train_set, batch_size=args.batchSize,
                                                       shuffle=True, num_workers=2, collate_fn=pad_packed_collate)
        bird_test_set = BirdTestSet(test_dir=test_dir, root_dir=root_dir, transform=transform_test)
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
        img = transforms.ToTensor()(Image.open(os.path.expanduser(name)))
        label = self.labels.iloc[idx]
        item = {'image': img, 'label': label}  # change this to labels?
        return item


class BirdTestSet(torch.utils.data.Dataset):

    def __init__(self, test_dir, root_dir, transform):
        self.test_dir = test_dir
        self.root_dir = root_dir
        names = []
        for root, directories, files in os.walk(root_dir + test_dir):
            for filename in files:
                names.append(filename)
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    # Not given labels for test data (because it's a competition)
    def __getitem__(self, idx):
        name = os.path.join(self.root_dir, self.names[idx])
        img = transforms.ToTensor()(Image.open(os.path.expanduser(name)))
        return img

# Padding function from:
# https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(audio, target)].
             audio is a FloatTensor
             target is a LongTensor with a length of 8
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a.t(), b, a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
    packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    return packed_batch, labels

class CollateWithPadding:

    def __init__(self, dim=0):
        self.dim = dim

    def collate_with_padding(self, batch):
        max_len = max(map(lambda x: x[0].shape[self.dim], batch)) # longest one
        batch = map(lambda x, y: (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.collate_with_padding(batch)