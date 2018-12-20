import numpy as np
import pandas as pd
import datetime
import copy

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils import model_zoo

import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Augmentation
# Not include RandomDegree since this seems to cause slow training accuracy advance
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(
                                brightness=0.1*torch.randn(1),
                                contrast=0.1*torch.randn(1),
                                saturation=0.1*torch.randn(1),
                                hue=0.1*torch.randn(1)),
                               transforms.ToTensor(),
				transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

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

model = resnet18().to(device)
model2 = copy.deepcopy(model)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(),lr=1e-5)

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
        outputs = model2(images)
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
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model2(images)
            loss_test = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (labels == predicted).sum().item()

        test_acc_vec[epoch] = correct_test / total_test * 100
        print("Test Accuracy of the model on test images:{}%".format(test_acc_vec[epoch]))

    print("Total cost time:{}".format(now_time-start_time))
            
# Save model
torch.save(model,"ResNetwithPretrain.ckpt")
