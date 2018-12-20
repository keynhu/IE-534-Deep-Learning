# This py file reads train_embed.h5 and val_embed.h5 and do neighbor selection only

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import h5py
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

# Classification model
neigh = NearestNeighbors(n_neighbors=30, n_jobs=8)

# Custom save accuracy
def save_checkpoint(obj, is_best, filename="checkpoint.pth.tar"):
    torch.save(obj, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)

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

model = torch.load("best_ImageRankingResNet.ckpt")
with torch.no_grad():
    with h5py.File("HW5/train_embed.h5","r") as f:
        train_embed = f["Train_Embed"][:]
    with h5py.File("HW5/val_embed.h5","r") as f:
        val_embed = f["Val_Embed"][:]
    # For each validation image
    count_correct = 0
    totalcount = len(val_loader)

    neigh.fit(train_embed.tolist())
    print("Neighbors of train embeddings fitted!")
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time - start_time)
    neighbors = neigh.kneighbors(val_embed, return_distance=False)
    neighbors_cls = neighbors // num_im # This only holds when training set is NOT shuffled
    print("Validation neighbors computed!")
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time - start_time)
    for j, (image,label) in enumerate(val_loader):
        line = neighbors_cls[j]
        count = Counter(line)
        mx = np.max(list(count.values()))
        locmx = [x for x,val in enumerate(count.values()) if val==mx]
        arglabel = np.array(list(count.keys()))[locmx]
        if len(arglabel) > 1:
            arglabel = np.random.choice(arglabel, size=1)
        if arglabel == label:
            count_correct += 1
        if (j+1)%1000==0:
            print("Current Process: {}/{}, Accuracy: {}%".format(j+1, totalcount, count_correct/totalcount * 100))
