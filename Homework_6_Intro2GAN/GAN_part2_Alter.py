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
model = torch.load('HW6/best_GAN_Baseline_Discriminator.ckpt').to(device)
model.eval()
# Load a batch of images, alter the labels
batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).to(device)
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).to(device)
Y_batch = Variable(Y_batch).to(device)
# Plot the real images
samples = X_batch.data.cpu().numpy()
samples += 1
samples /= 2
samples = samples.transpose(0,2,3,1)
fig = custom_plot(samples[0:100])
plt.savefig("HW6/Visualization/real_images.png", bbox_inches="tight")
plt.close(fig)
# Compute the prediction accuracy
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)
# Compute and plot gradients
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)
gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).to(device),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]
## Save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image)) # Normalization
gradient_image = gradient_image.transpose(0,2,3,1)
fig = custom_plot(gradient_image[0:100])
plt.savefig('HW6/Visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)
# Plot the fake images
## jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0
gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0
## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)
## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)
fig = custom_plot(samples[0:100])
plt.savefig('HW6/Visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)
