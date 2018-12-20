# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy
import re
import shutil

import torch
import torch.nn as nn
import torch.utils.data as Data

import torchvision
from torch.utils import model_zoo
import torchvision.transforms as transforms

import skimage
from skimage import transform as tf
from skimage import io

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_tp = 200 # Number of classes
num_im = 500 # Number of images in each class
total_im = num_tp * num_im
num_epochs = 20
batch_size = 64
total_step = total_im // batch_size + 1
learning_rate = 1e-3
embedding_size = 4096

# Build model
# Problem: How to substitute the last linear layer so that embedding vector instead of classification is returned
# Define pre-trained model ResNet18
def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',model_dir="./"))
    return model

mod = resnet18()
model = copy.deepcopy(mod)
model.fc = nn.Linear(512, embedding_size, bias=True) # Adapt for our model
model = model.to(device)
print("Pretrained model Loaded!")

# Build criterion and optimizer
criterion = nn.TripletMarginLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=0.9)

# Custom save accuracy
def save_checkpoint(obj, is_best, filename="checkpoint.pth.tar"):
    torch.save(obj, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)

# Define custom Dataset class
class TinyImageDataset(Data.Dataset):
    def __init__(self, Impath, label=None, train=True):
        self.path = Impath
        self.label = label
        self.train = train
    
    def __getitem__(self, index):
        """
        Args:
        index (int): Index of Dataset
        
        Return:
        if train: 
            img_triplet (np.array, 3 x 224 x 224 x 3)
        else: (Will return error if label==None and train==False)
            img (np.array, 224 x 224 x 3), label (str) 
        """
        if self.train:
            img_triplet_path = re.split(string=self.path[index], pattern="\t")
            img_triplet = [np.reshape(io.imread(img_triplet_path[i]), newshape=(3,224,224)) for i in range(3)]
            return img_triplet
        else:
            img = io.imread(self.path[index])
            label = self.label[index]
            return img, label
    
    def __len__(self):
        return len(self.path)
    
    def __repr__(self):
        return "Triplet_Generate_TinyImageNet200"

# In each epoch, zip a new trainset and train the network
start_time = datetime.datetime.now()
print("Start Time: ", start_time)

ave_loss_total_seq = []
for epoch in range(num_epochs):
    # Load new dataset
    file1 = open("HW5/ImagePath/Epoch_{}_Triplet.txt".format(epoch+1), "r")
    tm = file1.readlines()
    train_dataset = TinyImageDataset(Impath = tm)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    # Run model
    loss_total = 0 # Total loss in each epoch
    for i, image_triplet in enumerate(train_loader):
        #print(image_triplet[0].shape, image_triplet[1].shape, image_triplet[2].shape)
        image_qr = image_triplet[0].to(device).float()
        image_ps = image_triplet[1].to(device).float()
        image_ng = image_triplet[2].to(device).float()
            
        # Forward
        embed_qr = model(image_qr)
        embed_ps = model(image_ps)
        embed_ng = model(image_ng)
        loss = criterion(embed_qr, embed_ps, embed_ng)
        #total += image_qr.size(0)
        #correct += (loss==0).sum().item()
        loss_total += loss.item()
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%200 == 0:
            #train_acc = correct / total
            print("Epoch[{}/{}], Step[{}/{}], Total Loss {:4f}".format(
            epoch+1, num_epochs, i+1, total_step, loss_total))
            now_time = datetime.datetime.now()
            print("Cost Time: ", now_time - start_time)
        
    #train_acc = correct / total
    #train_acc_seq.append(train_acc)
    ave_loss_total = loss_total / total_im
    ave_loss_total_seq.append(ave_loss_total)
    is_best = (ave_loss_total == np.min(ave_loss_total_seq))
    print("Epoch[{}/{}], Average Loss {:4f}".format(
            epoch+1, num_epochs, ave_loss_total))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time - start_time)

    state = {"Epoch": epoch, "Train_Loss": ave_loss_total_seq}
    save_checkpoint(state, is_best, filename="checkpoint.pth.tar")
    save_checkpoint(model, is_best, filename="ImageRankingResNet.ckpt")
