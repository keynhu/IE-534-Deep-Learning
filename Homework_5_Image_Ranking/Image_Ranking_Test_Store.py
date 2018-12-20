# This py file stores train embeddings and validation embeddings under the best model

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py
import datetime
import shutil

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

from collections import Counter
from sklearn.neighbors import NearestNeighbors

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_tp = 200 # Number of classes
num_im = 500 # Number of images in each class
total_im = num_tp * num_im
embedding_size = 4096

# Load validation set
valfolder = "HW5/tiny-imagenet-200/val/images/"
valdataset = torchvision.datasets.ImageFolder(valfolder, transform=transforms.ToTensor()) 
val_loader = Data.DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=8)
# Load training set for convenience of embedding
trainfolder = "HW5/tiny-imagenet-200/train"
train_embed_dataset = torchvision.datasets.ImageFolder(trainfolder, transform=transforms.ToTensor()) 
train_embed_loader = Data.DataLoader(train_embed_dataset, batch_size=1, shuffle=False, num_workers=8)  
print("Datasets Loaded!")

# In each epoch, zip a new trainset and train the network
start_time = datetime.datetime.now()
print("Start Time: ", start_time)

example = np.random.randn(100000,4096)
with h5py.File("HW5/example.h5") as f:
    # create a dataset for your movie
    dst = f.create_dataset("random", data = example)
print("Example saved!")

model = torch.load("best_ImageRankingResNet.ckpt")
with torch.no_grad():
    train_embed = np.empty([total_im, embedding_size])
    # First, compute training embeddings
    for i, (image,label) in enumerate(train_embed_loader):
        image = image.to(device).float()
        image_embed = model(image)
        train_embed[i,:] = image_embed
        if (i+1)%10000==0:
            print("Train Embedding: {}/{}".format(i+1, total_im))
            now_time = datetime.datetime.now()
            print("Cost time: {}".format(now_time-start_time))
    print("Train Embeddings computed!")

with h5py.File("HW5/train_embed.h5") as f:
    dst = f.create_dataset("Train_Embed", data = train_embed)
print("Training Embeddings saved!")

total_im_val = len(val_loader)
with torch.no_grad():
    val_embed = np.empty([total_im, embedding_size])
    # First, compute training embeddings
    for j, (image,label) in enumerate(val_loader):
        image = image.to(device).float()
        image_embed = model(image)
        val_embed[i,:] = image_embed
        if (j+1)%1000==0:
            print("Validation Embedding: {}/{}".format(j+1, total_im_val))
            now_time = datetime.datetime.now()
            print("Cost time: {}".format(now_time-start_time))
    print("Validation Embeddings computed!")

with h5py.File("HW5/val_embed.h5") as f:
    dst = f.create_dataset("Val_Embed", data = val_embed)
print("Validation Embeddings saved!")
