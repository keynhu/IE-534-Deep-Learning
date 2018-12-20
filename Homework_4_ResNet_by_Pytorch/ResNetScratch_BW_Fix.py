import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.utils.data as Data

import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Augmentation
# Not include RandomDegree since this seems to cause slow training accuracy advance
ds_trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(
                                brightness=0.1*torch.randn(1),
                                contrast=0.1*torch.randn(1),
                                saturation=0.1*torch.randn(1),
                                hue=0.1*torch.randn(1)),
                              transforms.ToTensor()])

# Load data
BatchSize = 100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                        transform=ds_trans)
train_loader = Data.DataLoader(trainset, batch_size=BatchSize, shuffle=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                       transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False)

# Define residual block
class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample # (Conv2d + BatchNorm2d) that reduces the size of sample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# Define ResNet
class ResNet(nn.Module):
    def __init__(self, block, channel_first, channel_per_layer, block_per_layer, stride_per_layer, 
                 dropout_rate=0.25, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = channel_first
        self.conv = nn.Conv2d(in_channels=3, out_channels=channel_first, kernel_size=3, padding=1) # Inputs are 3-dim images
        self.bn = nn.BatchNorm2d(channel_first)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        if not (len(block_per_layer)==len(channel_per_layer)==len(stride_per_layer)):
            raise ValueError("Length of parameters must be the same!") 
        self.num_layer = len(block_per_layer)
        for i in range(self.num_layer):
            exec("self.layer{} = self.make_layer(block=block, out_channel={}, num_block={}, stride={})".format(
            i+1, channel_per_layer[i], block_per_layer[i], stride_per_layer[i]))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(int(32/np.prod(stride_per_layer)/2)**2 * channel_per_layer[-1], num_classes)
        
    def make_layer(self, block, out_channel, num_block, stride):
        downsample = None
        if (stride!=1) or (self.in_channels != out_channel):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel))
        layers = []
        layers.append(block(self.in_channels, out_channel, stride=stride, downsample=downsample))
        self.in_channels = out_channel
        for i in range(1, num_block):
            layers.append(block(out_channel, out_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = eval("(".join(["self.layer{}".format(i) for i in range(self.num_layer,0,-1)])+"(out"+")"*self.num_layer)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Run CIFAR100
# Build model
model = ResNet(block=Residual_block, channel_first=32, channel_per_layer=[32,64,128,256], 
               block_per_layer=[2,4,4,2], stride_per_layer=[1,2,2,2],dropout_rate=0.5).to(device)
# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

# Model training process
start_time = datetime.datetime.now()
num_epochs = 50
num_steps = len(train_loader)
train_acc_vec = np.zeros(num_epochs)
test_acc_vec = np.zeros(num_epochs)

for epoch in range(num_epochs):
    total = 0
    correct = 0 # Count accuracy in each epoch
    if (epoch > 5):
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
            
    train_acc_vec[epoch] = correct / total * 100
    print("Epoch[{}/{}], Step[{}/{}], Loss {:4f}, Accuracy {:4f}%".format(
        epoch+1, num_epochs, i+1, num_steps, loss.item(), train_acc_vec[epoch]))
    now_time = datetime.datetime.now()

    # Evaluate test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss_test = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (labels == predicted).sum().item()

    test_acc_vec[epoch] = correct_test / total_test * 100
    print("Test Accuracy of the model on test images:{}%".format(test_acc_vec[epoch]))
    print("Total cost time:{}".format(now_time-start_time))

# Save model
torch.save(model,"ResNetScratch.ckpt")
