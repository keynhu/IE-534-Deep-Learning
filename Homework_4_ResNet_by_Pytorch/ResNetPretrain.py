import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy
import sys

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils import model_zoo

import torchvision
from torchvision import transforms

# Create a logging file
old_stdout = sys.stdout
log_file = open("message2.log","w")
sys.stdout = log_file

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Augmentation
# Not include RandomDegree since this seems to cause slow training accuracy advance
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
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
                                       transform=ds_trans)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False)

# Define pre-trained model ResNet18
def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',model_dir="./"))
    return model

model = resnet18()
model2 = copy.deepcopy(model)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

# Model training process
start_time = datetime.datetime.now()
num_epochs = 50
num_steps = len(train_loader)

for epoch in range(num_epochs):
    total = 0
    correct = 0 # Count accuracy in each epoch
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
            
        # Forward
        outputs = model2(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    print("Epoch[{}/{}], Step[{}/{}], Loss {:4f}, Accuracy {:4f}%".format(
        epoch+1, num_epochs, i+1, num_steps, loss.item(), correct/total*100))
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
torch.save(model,"ResNetwithPretrain.ckpt")

# Close log file
sys.stdout = old_stdout
log_file.close()
