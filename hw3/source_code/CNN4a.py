import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
import time
import datetime

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(model_weights['conv1.weight']):
        # print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')

    plt.show()


def plot_kernel_output(model, images):
    fig1 = plt.figure()
    plt.figure(figsize=(1, 1))

    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1, 2, 0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()


def test_accuracy(net, dataloader):
    ########TESTING PHASE###########

    # check accuracy on whole test set
    correct = 0
    total = 0
    net.eval()  # important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
        accuracy))
    return accuracy


def plot_with_epoch(filename, sup_title, acc, loss, ep):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    z = range(1, ep + 1)
    color = 'tab:blue'
    plt.xticks(z)
    fig.suptitle(sup_title, fontsize=14, fontweight='bold')
    ax1.set_title('After %d Epochs - Loss = %2.3f - Accuracy = %2.0f%%' % (ep, loss[ep - 1], acc[ep - 1]))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.plot(z, acc, '-o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel("Epoch")
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(z, loss, '-o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    # left, right = plt.xlim()
    # down, up = plt.ylim()
    # r_x = (right - left)/100
    # r_y = ((up - down)/50)*plus
    # for i, txt in enumerate(acc):
    #     ax.annotate(("%.3f" % txt), (z[i], acc[i]), (z[i] + r_x , acc[i] + r_y), fontsize=9)
    plt.savefig(filename, transparent=False, dpi=150, bbox_inches="tight")
    fig.show()


n_classes = 100


# function to define an old style fully connected network (multilayer perceptrons)
class old_nn(nn.Module):
    def __init__(self):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)  # last FC for classification

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# function to define the convolutional network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv_final_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.pool(self.conv_final_bn(self.conv_final(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#### RUNNING CODE FROM HERE:


start = time.time()

# transform are heavily used to do simple and complex transformation and data augmentation
transform_train = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        # transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
if torch.cuda.is_available():
    print("CUDA available")
else:
    print("CUDA unavailable")
    exit(-1)
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4, drop_last=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=4, drop_last=True)

dataiter = iter(trainloader)

###OPTIONAL:
# show images just to understand what is inside the dataset ;)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
####

# create the old style NN network
# net = old_nn()
###

net = CNN()
###
# for Residual Network:
# net = models.resnet18(pretrained=True)
# net.fc = nn.Linear(512, n_classes)  # changing the fully connected layer of the already allocated network
####

###OPTIONAL:
# print("####plotting kernels of conv1 layer:####")
# plot_kernel(net)
####

net = net.cuda()

criterion = nn.CrossEntropyLoss().cuda()  # it already does softmax computation for use!
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # better convergency w.r.t simple SGD :)

###OPTIONAL:
# print("####plotting output of conv1 layer:#####")
# plot_kernel_output(net,images)
###

########TRAINING PHASE###########
n_loss_print = len(trainloader)  # print every epoch, use smaller numbers if you wanna print loss more often!
n_epochs = 20
losses = np.empty(n_epochs)
accuracies = np.empty(n_epochs)
for epoch in range(n_epochs):  # loop over the dataset multiple times
    net.train()  # important for activating dropout and correctly train batchnorm
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs and cast them into cuda wrapper
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % n_loss_print == (n_loss_print - 1):
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / n_loss_print))
            losses[epoch] = running_loss / n_loss_print
            running_loss = 0.0

    accuracies[epoch] = test_accuracy(net, testloader)
    end = time.time()
    delta = int(end - start)
    print("Time elapsed: " + str(datetime.timedelta(seconds=delta)))
print('Finished Training')

plot_with_epoch("fig04a.png", "Step 4/6 - CNN 128/128/128/256 BN", accuracies, losses, n_epochs)

end = time.time()
delta = int(end - start)
print(delta)
print(str(datetime.timedelta(seconds=delta)))
