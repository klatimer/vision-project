from utils import argParser
from dataloader import BirdLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import pdb

from torch.autograd import Variable
import datetime
import time
import pandas as pd
import os

# Cuda / CPU setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

        outputs = net(inputs)

        # backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 200 == 0:
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
        if tag == 'Train':
            for data in dataTestLoader:
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            net.log('%s Accuracy of the network: %d %%' % (tag,
                100 * correct / total))

        else: # Need to write a csv file of predictions to submit to kaggle
            dfs = []

            filename = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_predictions.csv')
            prediction_dir = 'predictions'
            if not os.path.exists(prediction_dir):
                os.mkdir(prediction_dir)

            for data in dataTestLoader:
                inputs, names = data['image'], data['name']
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predictions = torch.max(outputs.data, 1)
                for i in range(len(names)):
                    # dfs.append(pd.DataFrame([[names[i]], [predictions[i].item()]], columns=('path', 'class')))
                    dfs.append(pd.DataFrame({'path': names[i], 'class': predictions[i].item()}, index=[i]))
                    # dfs.append(pd.DataFrame.from_items(('path', names[i]), ('class', predictions[i].item())))

            df = pd.concat(dfs).reindex(columns=['path', 'class'])
            df.to_csv(os.path.join(prediction_dir, filename), encoding='utf-8', index=False)


    """
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

    
    for i in range(555):
        net.log('%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))
    """


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

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for epoch in range(args.epochs):
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, birdLoader, optimizer, criterion, epoch)
        if epoch % 5 == 0: # Log training accuracy every 5 epochs
            test(net, birdLoader, 'Train')
        if epoch % 10 == 0: # write csv output every 10 epochs and save model
            test(net, birdLoader, 'Test')
            try:
                torch.save(net, os.path.join(model_dir, 'model.pt'))
            except:
                print("Could not save model")

    # Save the model
    torch.save(net, 'model.pt')

    print('The log is recorded in ')
    print(net.logFile.name)

if __name__ == '__main__':
    main()