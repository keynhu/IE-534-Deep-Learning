# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import shutil

import torch
import torch.nn as nn
import torch.utils.data as Data

import torchvision
from torch.utils import model_zoo
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
BatchSize = 128
learning_rate = 1e-4

# Save function
def save_checkpoint(obj, is_best, filename="checkpoint.pth.tar"):
    torch.save(obj, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transform_train)
train_loader = Data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=8)
        
# Build model
# Create torch model with structure given in reference
LongConv = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.LayerNorm(normalized_shape=(196,32,32)),
                         nn.LeakyReLU(inplace=True), # Conv layer 1
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=2,padding=1),
                         nn.LayerNorm(normalized_shape=(196,16,16)),
                         nn.LeakyReLU(inplace=True), # Conv layer 2
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1), 
                         nn.LayerNorm(normalized_shape=(196,16,16)),
                         nn.LeakyReLU(inplace=True), # Conv layer 3
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=2,padding=1),
                         nn.LayerNorm(normalized_shape=(196,8,8)),
                         nn.LeakyReLU(inplace=True), # Conv layer 4
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.LayerNorm(normalized_shape=(196,8,8)),
                         nn.LeakyReLU(inplace=True), # Conv layer 5
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.LayerNorm(normalized_shape=(196,8,8)),
                         nn.LeakyReLU(inplace=True), # Conv layer 6
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.LayerNorm(normalized_shape=(196,8,8)),
                         nn.LeakyReLU(inplace=True), # Conv layer 7
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=2,padding=1),
                         nn.LayerNorm(normalized_shape=(196,4,4)),
                         nn.LeakyReLU(inplace=True), # Conv layer 8
                         nn.MaxPool2d(kernel_size=4,stride=4)) # Max Pooling                       
Scorer = nn.Linear(in_features=196,out_features=1,bias=True)
Classifier = nn.Linear(in_features=196,out_features=10,bias=True)
# Define a Convolution NN class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ConvLayer = LongConv
        self.Class = Classifier
        self.Score = Scorer
        
    def forward(self,x):
        ConvOut = self.ConvLayer(x)
        Out = ConvOut.reshape(ConvOut.shape[0],-1)
        ScoreOut = self.Score(Out)
        ClassOut = self.Class(Out)
        return ScoreOut, ClassOut
# Define model
model = ConvNet().to(device)
# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# In each epoch
# Training and testing process
start_time = datetime.datetime.now()
train_acc_seq = []
test_acc_seq = []

for epoch in range(num_epochs):
    total_train = 0
    correct_train = 0
    if epoch == 50:
        for group in optimizer.param_groups:
            group['lr'] = learning_rate / 10
    if epoch == 75:
        for group in optimizer.param_groups:
            group['lr'] = learning_rate / 100
    if epoch > 5:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
        if labels.shape[0] < BatchSize:
            continue
        # Forward
        _, classes = model(images)
        loss = criterion(classes, labels)
        _, predicted = torch.max(classes.data, 1)
        total_train += labels.size(0)
        correct_train += (labels == predicted).sum().item()
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Compute train loss and accuracy
    train_acc = correct_train/total_train
    train_acc_seq.append(train_acc)
    print("Epoch[{}/{}], Accuracy {:4f}%".format(
                epoch+1, num_epochs, train_acc*100))
    now_time = datetime.datetime.now()
    print("Total cost time:{}".format(now_time-start_time))
    
    # Compute test loss and accuracy
    correct_test = 0
    total_test = 0
    model.eval()
    with torch.no_grad():
        for j, (images,labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).long()
            
            _, outputs = model(images)
            loss_test = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (labels == predicted).sum().item()
    test_acc = correct_test/total_test
    test_acc_seq.append(test_acc)
    is_best = (test_acc == np.max(test_acc_seq))
    print("Test Accuracy of the model on test images:{}%".format(test_acc*100))
    now_time = datetime.datetime.now()
    print("Total cost time:{}".format(now_time-start_time))
    
    # Save model and checkpoints
    state = {"Epoch": epoch, "Train_Accuracy": train_acc_seq, "Test_Accuracy": test_acc_seq}
    save_checkpoint(state, is_best=False, filename="checkpoint.pth.tar")
    save_checkpoint(model, is_best, filename="GAN_Baseline_Discriminator.ckpt")
