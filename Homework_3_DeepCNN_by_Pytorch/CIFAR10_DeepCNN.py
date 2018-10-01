import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import datetime

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms

# Data augmentation
ds_trans = transforms.Compose([
                              transforms.RandomVerticalFlip(),
                              transforms.RandomCrop(32),
                              transforms.ToTensor()])

# Load data
BatchSize = 100
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=ds_trans)
train_loader = Data.DataLoader(trainset, batch_size=BatchSize, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=ds_trans)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create torch model with structure given in Slides "Lecture 6"
DropRate = 0.1
LongConv = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=1,padding=2),
                         nn.ReLU(inplace=True), # Conv layer 1
                         nn.BatchNorm2d(num_features=64),
                         nn.Conv2d(64,64,4,padding=2),
                         nn.ReLU(inplace=True), # Conv layer 2
                         nn.MaxPool2d(kernel_size=2,stride=2), # Max Pooling
                         nn.Dropout2d(p=DropRate), # Dropout
                         nn.Conv2d(64,64,4,padding=2), 
                         nn.ReLU(inplace=True), # Conv layer 3
                         nn.BatchNorm2d(64),
                         nn.Conv2d(64,64,4,padding=2),
                         nn.ReLU(inplace=True), # Conv layer 4
                         nn.MaxPool2d(kernel_size=2,stride=2), # Max Pooling
                         nn.Dropout2d(p=DropRate), # Dropout
                         nn.Conv2d(64,64,4,padding=2), 
                         nn.ReLU(inplace=True), # Conv layer 5
                         nn.BatchNorm2d(64),
                         nn.Conv2d(64,64,3),
                         nn.ReLU(inplace=True), # Conv layer 6
                         nn.Dropout2d(p=DropRate), # Dropout
                         nn.Conv2d(64,64,3),
                         nn.ReLU(inplace=True), # Conv layer 7
                         nn.BatchNorm2d(64),
                         nn.Conv2d(64,64,3),
                         nn.ReLU(inplace=True), # Conv layer 8
                         nn.BatchNorm2d(64),
                         nn.Dropout2d(p=DropRate)) # Dropout
Classifier = nn.Sequential(
                         nn.Linear(in_features=64*4*4,out_features=500,bias=True),
                         nn.Linear(in_features=500,out_features=500,bias=True),
                         nn.Linear(in_features=500,out_features=10,bias=True),
                         nn.Softmax(dim=1))

# Define a Convolution NN class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ConvLayer = LongConv
        self.LinearLayer = Classifier
        
    def forward(self,x):
        ConvOut = self.ConvLayer(x)
        Out = ConvOut.reshape(ConvOut.shape[0],-1)
        ClassOut = self.LinearLayer(Out)
        return ClassOut

# Define model
model = ConvNet().to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

# Training process
start_time = datetime.datetime.now()
num_epochs = 100
num_steps = len(train_loader)

for epoch in range(num_epochs):
    total = 0
    correct = 0
    if(epoch>6):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if(state['step']>=1024):
                    state['step'] = 1000

    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
            
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch[{}/{}], Loss {:4f}, Accuracy {:4f}%".format(
            epoch+1, num_epochs, loss.item(), correct/total*100))
    now_time = datetime.datetime.now()
    print("Total cost time:{}".format(now_time-start_time))

# Evaluate with test set
model.eval()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        
        outputs = model(images)
        loss_test = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (labels == predicted).sum().item()
    
    print("Test Accuracy of the model on test images:{}%".format(correct_test / total_test * 100))
    
# Save model
torch.save(model,"ConvModel.ckpt")
