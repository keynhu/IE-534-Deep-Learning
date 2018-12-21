# Import modules
import os
import sys
import numpy as np
import pandas as pd
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchvision
# import torchvision.transforms as transforms

from multiprocessing import Pool
from util_AR import save_checkpoint, getUCF101, loadFrame

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 1e-4
num_of_epochs = 50

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data
data_dir = "/projects/training/bauh/AR/"
class_list, train, test = getUCF101(data_dir)

# Introduce pretrained ResNet-50 model
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, NUM_CLASSES)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad_(False)
    
# Activate some parameters for training
params = []
# for param in model.conv1.parameters():
#     param.requires_grad_(True)
#     params.append(param)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
#     params.append(param)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
#     params.append(param)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
#     params.append(param)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
#     params.append(param)
for param in model.layer4[2].parameters():
    param.requires_grad_(True)
    params.append(param)
for param in model.fc.parameters():
    param.requires_grad_(True)
    params.append(param)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Activate multiple CPU cores
pool_threads = Pool(8, maxtasksperchild=200)

epoch_train_acc = []
epoch_test_acc = []
start_time = datetime.datetime.now()
print("Start training at: ", start_time)

for epoch in range(num_of_epochs):
    ## TRAIN
    train_acc = []
    model.train()
    random_indices = np.random.permutation(len(train[0])) # Random indices of batches
    for i in range(0, len(train[0])-batch_size, batch_size):
        augment = True
        video_list = [(train[0][k], augment) for k in random_indices[i:i+batch_size]]
        data = pool_threads.map(loadFrame, video_list)
        
        next_batch = 0
        for video in data: # For each video in video_list (i.e. the current batch)
            if video.size == 0: # If there is an empty video, skip the whole batch
                next_batch = 1
        if next_batch:
            continue
        
        x = np.asarray(data, dtype=np.float32)
        x = torch.FloatTensor(x).to(device).contiguous()
        y = train[1][random_indices[i:i+batch_size]]
        y = torch.from_numpy(y).to(device)
        
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = float(prediction.eq(y.data).sum()) / float(batch_size) * 100.0
        train_acc.append(accuracy)
    
    ave_train_acc = np.mean(train_acc)
    epoch_train_acc.append(ave_train_acc)
    print("Epoch {}/{}, cumulative train accuracy: {}%".format(epoch+1, num_of_epochs, ave_train_acc))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    ## TEST
    model.eval()
    test_acc = []
    random_indices2 = np.random.permutation(len(test[0]))
    for i in range(0, len(test[0])-batch_size, batch_size):
        augment = False
        video_list2 = [(test[0][k], augment) for k in random_indices2[i:i+batch_size]]
        data2 = pool_threads.map(loadFrame, video_list2)
        
        next_batch = 0
        for video in data: # For each video in video_list (i.e. the current batch)
            if video.size == 0: # If there is an empty video, skip the whole batch
                next_batch = 1
        if next_batch:
            continue
        
        x = np.asarray(data2, dtype=np.float32)
        x = torch.FloatTensor(x).to(device).contiguous()
        y = test[1][random_indices2[i:i+batch_size]]
        y = torch.from_numpy(y).to(device)
        
        output = model(x)
        
        prediction = output.data.max(1)[1]
        accuracy = float(prediction.eq(y.data).sum()) / float(batch_size) * 100.0
        test_acc.append(accuracy)
    
    ave_test_acc = np.mean(test_acc)
    epoch_test_acc.append(ave_test_acc)
    print("Epoch {}/{}, cumulative test accuracy: {}%".format(epoch+1, num_of_epochs, ave_test_acc))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    is_best = (ave_test_acc == np.max(epoch_test_acc))
    save_checkpoint(model, is_best, "Singleframe_Video_AR.ckpt")
    dic = {"Epoch": epoch, "Train_Accuracy": epoch_train_acc, "Test_Accuracy": epoch_test_acc}
    torch.save(dic, "Singleframe_Video_AR.checkpoint.pth.tar")

pool_threads.close()
pool_threads.terminate()
