# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import shutil
import h5py

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
neigh = NearestNeighbors(n_neighbors=30, n_jobs=32)

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
model.eval()
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

    neigh.fit(train_embed.tolist())
    # Then, for each validation image
    count_correct = 0
    totalcount = len(val_loader)
    val_acc_seq = []
    neighbors = []
    val_embed = np.empty([totalcount, embedding_size])
    for j, (image,label) in enumerate(val_loader):
        image = image.to(device).float()
        image_embed = model(image)
        val_embed[j,:] = image_embed
        if (j+1)%1000==0:
            print("Valid Embedding: {}/{}.".format(j+1, totalcount))
            now_time = datetime.datetime.now()
            print("Cost time: {}".format(now_time-start_time))        
    print("Validation Embeddings computed!")

    with h5py.File("HW5/val_embed.h5") as f:
        dst = f.create_dataset("Val_Embed", data = val_embed)
    print("Validation Embeddings saved!")

    # neighbors = neigh.kneighbors(val_embed, return_distance=False)
    for j in range(100):
        neighbors.append(neigh.kneighbors(val_embed[j*100:(j+1)*100,:], return_distance=False))
        print("Neighbors Computed: ", (j+1)*100)
        now_time = datetime.datetime.now()
        print("Cost Time: ", now_time-start_time)
    neighbors = np.array(neighbors)
    neighbors_cls = neighbors // num_im # This only holds when training set is NOT shuffled
    neighbors_cls = np.reshape(neighbors_cls, (totalcount, 30))

    with h5py.File("HW5/val_neighbors.h5") as f:
        dst = f.create_dataset("Val_Neighbors", data = neighbors_cls)
    print("Validation neighbors computed!")

    for j, (image,label) in enumerate(val_loader):
        line = neighbors_cls[j,:]
        count = Counter(line)
        mx = np.max(list(count.values()))
        locmx = [x for x,val in enumerate(count.values()) if val==mx]
        arglabel = np.array(list(count.keys()))[locmx]
        if len(arglabel) > 1:
            count_correct += (label in arglabel)
        elif arglabel == label:
            count_correct += 1
        if (j+1)%1000==0:
            val_acc = count_correct/totalcount
            val_acc_seq.append(val_acc)
            print("Current Process: {}/{}, Accuracy: {}%".format(j+1, totalcount, val_acc * 100))
            now_time = datetime.datetime.now()
            print("Cost Time: ", now_time-start_time)

state = {"Val_Accuracy": val_acc_seq}
save_checkpoint(state, filename="val_acc_checkpoint.pth.tar")
