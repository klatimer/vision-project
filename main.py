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
        inputs, labels = data['image'], data['label']
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 500 == 0:    # print every 2000 mini-batches
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
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += data['label'].size(0)
            correct += (predicted == labels).sum().item()

    net.log('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = [0.] * 555
    class_total = [0.] * 555
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == labels).squeeze()

            for i in range(len(data['label'])):
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
    print(birdLoader.classes[len(birdLoader.classes) - 1])

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
            test(net, birdLoader, 'Test')

    print('The log is recorded in ')
    print(net.logFile.name)

if __name__ == '__main__':
    main()