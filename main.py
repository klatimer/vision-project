from utils import argParser
from dataloader import BirdLoader, BirdDemoLoader
import torch
import datetime
import time
from PIL import Image, ImageDraw, ImageFont, ImageTk
import torchvision.transforms as transforms
import tkinter

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


def test(net, dataloader, tag='', save=None):

    correct = 0
    total = 0
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
        # dataTestLoader = dataloader.trainloader
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += data['label'].size(0)
            correct += (predicted == labels).sum().item()


    curAccuracy =  100 * correct / total
    net.log('%s Accuracy of the network: %d %%' % (tag,
        curAccuracy))
    if save is not None:
        if curAccuracy > test.prevAccuracy:
            print("Better model is saved")
            torch.save(net.state_dict(), save)
            test.prevAccuracy = curAccuracy

test.prevAccuracy = 0

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

def demo(net, dataloader):
    transform = transforms.ToPILImage()
    dataDemoLoader = dataloader.demoLoader

    window = tkinter.Tk()
    window.bind("<Button>", button_click_exit_mainloop)
    window.geometry('+%d+%d' % (100, 100))

    old_label_image = None
    with torch.no_grad():
        for data in dataDemoLoader:
            """ 
            Goes through each image, get the prediction from the mode,
            Compare it with the actual label, and show it on the screen
            """
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            if ((predicted == labels).sum().item()):
                result = "Correct"
            else:
                result = "Incorrect"

            """ Convert ogImg tensor data back to PIL image"""
            ogImg = transform(data['ogImg'].squeeze())

            """ Add some text on the image"""
            draw = ImageDraw.Draw(ogImg)
            font = ImageFont.truetype("/home/brandon/vision-project/fonts/arial.ttf", 20)
            draw.text((0, 0), "Predict: " + dataloader.classes[predicted] + " Actual: " + dataloader.classes[labels] + " -- " + result, (255, 255, 255), font=font)

            """ Using tkinter to show the image """
            window.geometry('%dx%d' % (ogImg.size[0], ogImg.size[1]))
            tkpi = ImageTk.PhotoImage(ogImg)
            label_image = tkinter.Label(window, image=tkpi)
            label_image.place(x=0, y=0, width=ogImg.size[0], height=ogImg.size[1])
            if old_label_image is not None:
                old_label_image.destroy()
            old_label_image = label_image
            window.mainloop()  # wait until user clicks the window

            ogImg.close()


def main():
    args = argParser()

    birdLoader = BirdLoader(args)
    argTrain = args.train
    argDemo = args.demo

    net = args.model()
    net = net.to(device)

    if argTrain is not None:
        """ Run training """
        print('The log is recorded in ')
        print(net.logFile.name)

        criterion = net.criterion().to(device)
        optimizer = net.optimizer()


        modelName = args.name
        modelPath = 'models/'
        ts = time.time()
        modelName = datetime.datetime.fromtimestamp(ts).strftime(modelName + '_%Y-%m-%d_%H:%M:%S.pt')
        modelName = modelPath + modelName


        for epoch in range(args.epochs):
            net.adjust_learning_rate(optimizer, epoch, args)
            train(net, birdLoader, optimizer, criterion, epoch)
            if epoch % 1 == 0: # Comment out this part if you want a faster training time
                test(net, birdLoader, 'Train')
                test(net, birdLoader, 'Test', modelName)

        print('The log is recorded in ')
        print(net.logFile.name)

    elif argDemo is not None:
        """ Run demo """
        modelPath = args.modelPath
        net.load_state_dict(torch.load(modelPath))  # Load the specified model from args
        demoLoader = BirdDemoLoader(args)           # Load the dataloader
        demo(net, demoLoader)                       # Run the demo



if __name__ == '__main__':
    main()