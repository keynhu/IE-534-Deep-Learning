# Import modules
import numpy as np
import pandas as pd
import datetime
import shutil

import torch
import torch.nn as nn
import torch.utils.data as Data

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

# Define a way to compute gradient
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(BatchSize, 1)
    alpha = alpha.expand(BatchSize, int(real_data.nelement()/BatchSize)).contiguous()
    alpha = alpha.view(BatchSize, 3, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BatchSize, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad = True

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# Custom reshape layer
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)

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

# Define model
genmodel = GenConvNet().to(device)
discmodel = DiscConvNet().to(device)
# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_g = torch.optim.Adam(genmodel.parameters(), lr=learning_rate, betas=(0,0.9))
optimizer_d = torch.optim.Adam(discmodel.parameters(), lr=learning_rate, betas=(0,0.9))

# In each epoch
# Training and testing process
# Complete process of training
start_time = datetime.datetime.now()
train_acc_seq = []
test_acc_seq = []

for epoch in range(num_epochs):
    # this avoids overflow
    if epoch > 5:
        for group in optimizer_g.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
    genmodel = genmodel.train()
    discmodel = discmodel.train()
    train_epoch_acc_seq = []
    for idx, (images, labels) in enumerate(train_loader):
        if labels.shape[0] < BatchSize:
            continue
        # train the generator
        for param in discmodel.parameters():
            param.requires_grad = False
        genmodel.zero_grad()
        ### Generate the input noise
        rand_label = np.random.randint(0, num_classes, BatchSize)
        noise = np.random.normal(0,1,(BatchSize,100))
        rand_label_onehot = np.zeros((BatchSize,num_classes))
        rand_label_onehot[np.arange(BatchSize), rand_label] = 1
        noise[np.arange(BatchSize), :num_classes] = rand_label_onehot[np.arange(BatchSize)]
        noise = noise.astype(np.float32)
        ts_noise = torch.from_numpy(noise).to(device) # Create noise as a Tensor
        fake_label = torch.from_numpy(rand_label).to(device) # Create fake label as a Tensor
        ### Train generator (i.e. Generate fake images, evaluate it by discriminator)
        fake_data = genmodel(ts_noise)
        gen_score, gen_class = discmodel(fake_data)
        gen_loss = criterion(gen_class, fake_label)
        gen_cost = -gen_score.mean() + gen_loss
        gen_cost.backward()
        optimizer_g.step()

        # train the discriminator with input from generator
        for param in discmodel.parameters():
            param.requires_grad = True
        discmodel.zero_grad()
        ### Generate fake images and evaluate
        with torch.no_grad():
            fake_data = genmodel(ts_noise)
        disc_fake_score, disc_fake_class = discmodel(fake_data)
        disc_fake_loss = criterion(disc_fake_class, fake_label)
        ### Train discriminator
        real_data = images.to(device)
        real_label = labels.to(device).long()
        disc_real_score, disc_real_class = discmodel(real_data)
        disc_real_loss = criterion(disc_real_class, real_label)
        prediction = disc_real_class.data.max(1)[1]
        accuracy = float(prediction.eq(real_label.data).sum()) / float(BatchSize)
        train_epoch_acc_seq.append(accuracy)
        grad_penalty = calc_gradient_penalty(discmodel, real_data, fake_data)
        disc_cost = disc_fake_score.mean() - disc_real_score.mean() + disc_real_loss + disc_fake_loss + grad_penalty
        disc_cost.backward()
        optimizer_d.step()
        if (idx+1) % 100 ==0:
            print("Current batch accuracy: ", accuracy)
            now_time = datetime.datetime.now()
            print("Cost Time: ", now_time-start_time)
    train_acc = np.mean(train_epoch_acc_seq)
    train_acc_seq.append(train_acc)
    print("Epoch: {}/{}, Training accuracy: {}%!".format(epoch+1, num_epochs, train_acc*100))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    # Compute test loss and accuracy
    correct_test = 0
    total_test = 0
    discmodel.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if labels.shape[0] < BatchSize:
                continue
            images = images.to(device)
            labels = labels.to(device).long()

            with torch.no_grad():
                _, output = discmodel(images)

            prediction = output.data.max(1)[1]
            correct_test += (labels == prediction).sum().item()
            total_test += labels.size(0)
    test_acc = correct_test / total_test
    test_acc_seq.append(test_acc)
    is_best = (test_acc == np.max(test_acc_seq))
    print("Test Accuracy of the model on test images:{}%".format(test_acc*100))
    now_time = datetime.datetime.now()
    print("Total cost time:{}".format(now_time-start_time))
    
    # Save model and checkpoints
    state = {"Epoch": epoch, "Train Accuracy": train_acc_seq, "Test_Accuracy": test_acc_seq}
    save_checkpoint(state, is_best=False, filename="GD_checkpoint.pth.tar")
    save_checkpoint(genmodel, is_best, filename="Generator.ckpt")
    save_checkpoint(discmodel, is_best, filename="Discriminator.ckpt")

    # Print fake images
    with torch.no_grad():
        genmodel.eval()
        samples = genmodel(ts_noise).cpu().numpy()
        samples += 1
        samples /= 2
        samples = samples.transpose(0,2,3,1)
        genmodel.train()

    fig = custom_plot(samples)
    plt.savefig("HW6/Plots/output_%s.png" % str(epoch).zfill(3), bbox_inches="tight")
    plt.close(fig)
