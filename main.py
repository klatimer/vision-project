from utils import argParser
from dataloader import BirdLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import pdb

from torch.autograd import Variable

# Cuda / CPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)


def train(net, dataloader, optimizer, criterion, epoch):
    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(dataloader.trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        if criterion == torch.nn.CrossEntropyLoss:
            loss = criterion(outputs, labels)
        else:
            # Convert labels to one-hot vector
            one_hot = torch.zeros(len(labels), 10).to(device)
            target = one_hot.scatter_(1, labels.data.unsqueeze(1), 1) # look along dim=1, replace indices from labels.data with 1
            target = Variable(target)
            loss = criterion(outputs, target.float())

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
            net.log('[%d, %5d] loss: %.9f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    net.log('Final Summary:   loss: %.3f' %
          (total_loss / i))


def test(net, dataloader, tag=''):

    correct = 0
    total = 0
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            labels = labels.to(device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.log('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data

            images = images.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            labels = labels.to(device)

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        net.log('%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

def main():
    args = argParser()

    birdLoader = BirdLoader(args)

    # Test the classes field
    print(birdLoader.classes[0])
    print(birdLoader.classes[1])

    # cifarLoader = CifarLoader(args)
    """
    net = args.model()
    net = net.to(device)

    print('The log is recorded in ')
    print(net.logFile.name)

    criterion = net.criterion().to(device)
    optimizer = net.optimizer()

    for epoch in range(args.epochs):
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, birdLoader, optimizer, criterion, epoch)
        if epoch % 1 == 0: # Comment out this part if you want a faster training time
            test(net, birdLoader, 'Train')

    print('The log is recorded in ')
    print(net.logFile.name)
    """

if __name__ == '__main__':
    main()