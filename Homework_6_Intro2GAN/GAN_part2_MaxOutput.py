# Import modules
import numpy as np
import pandas as pd
import datetime
import shutil

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 200
num_classes = 10
BatchSize = 100
learning_rate = 1e-4

# Data augmentation
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

# Custom reshape layer
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)

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

# Build model
# Create Generator model with structure given in reference
LongGenConv = nn.Sequential(nn.Linear(in_features=100, out_features=196*4*4, bias=True),
                            View(BatchSize,196,4,4),
                         nn.ConvTranspose2d(in_channels=196,out_channels=196,kernel_size=4,stride=2,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 1
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 2
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1), 
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 3
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 4
                         nn.ConvTranspose2d(in_channels=196,out_channels=196,kernel_size=4,stride=2,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 5
                         nn.Conv2d(in_channels=196,out_channels=196,kernel_size=3,stride=1,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 6
                         nn.ConvTranspose2d(in_channels=196,out_channels=196,kernel_size=4,stride=2,padding=1),
                         nn.BatchNorm2d(num_features=196),
                         nn.ReLU(inplace=True), # Conv layer 7
                         nn.Conv2d(in_channels=196,out_channels=3,kernel_size=3,stride=1,padding=1)) # Conv layer 8
# Create Discriminator model with structure given in reference
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

# Define a Generator class
class GenConvNet(nn.Module):
    def __init__(self):
        super(GenConvNet, self).__init__()
        self.GenLayer = LongGenConv
    def forward(self,x):
        return self.GenLayer(x)
# Define a Discriminator class
class DiscConvNet(nn.Module):
    def __init__(self):
        super(DiscConvNet, self).__init__()
        self.ConvLayer = LongConv
        self.Class = Classifier
        self.Score = Scorer
    def forward(self,x):
        ConvOut = self.ConvLayer(x)
        Out = ConvOut.reshape(ConvOut.shape[0],-1)
        ScoreOut = self.Score(Out)
        ClassOut = self.Class(Out)
        return ScoreOut, ClassOut

# Save function
def save_checkpoint(obj, is_best, filename="checkpoint.pth.tar"):
    torch.save(obj, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)

# Plot images
def custom_plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.imshow((sample*255).astype(np.uint8))
    return fig

# Load model
model1 = torch.load('HW6/best_GAN_Baseline_Discriminator.ckpt').to(device)
model1.eval()
model2 = torch.load('HW6/best_Discriminator.ckpt').to(device)
model2.eval()
# Load a batch of images, alter the labels
batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).to(device)
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).to(device)
Y_batch = Variable(Y_batch).to(device)
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)
Y = torch.arange(10).type(torch.int64).to(device)
# Train to maximize class output in model 1
lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model1(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).to(device),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)
    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0
# Plot the real images
samples = X.data.cpu().numpy()
samples += 1
samples /= 2
samples = samples.transpose(0,2,3,1)
fig = custom_plot(samples)
plt.savefig("HW6/Visualization/max_class_no_generator.png", bbox_inches="tight")
plt.close(fig)
# Train to maximize class output in model 2
lr = 0.1
weight_decay = 0.001
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)
Y = torch.arange(10).type(torch.int64).to(device)
for i in range(200):
    _, output = model2(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).to(device),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)
    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0
# Plot the real images
samples = X.data.cpu().numpy()
samples += 1
samples /= 2
samples = samples.transpose(0,2,3,1)
fig = custom_plot(samples)
plt.savefig("HW6/Visualization/max_class_with_generator.png", bbox_inches="tight")
plt.close(fig)
