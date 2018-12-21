# Import modules
import os
import sys
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torchvision
# import torchvision.transforms as transforms

from multiprocessing import Pool
from util_AR import save_checkpoint, getUCF101, loadSequence
import util_Resnet3d

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 16
lr = 1e-4
num_of_epochs = 20

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data
data_dir = "/projects/training/bauh/AR/"
class_list, train, test = getUCF101(data_dir)

# Introduce pretrained ResNet-50 model
model = util_Resnet3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)
pretrained = torch.load(data_dir + 'resnet-50-kinetics.pth')
pretrained_state_dict = {k[7:]: v.cpu() for k,v, in pretrained['state_dict'].items()}
model.load_state_dict(pretrained_state_dict)
model.fc = nn.Linear(model.fc.weight.shape[1], NUM_CLASSES)

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
for param in model.layer4[0].parameters(): # Ignore the other two Residual blocks in layer 4
    param.requires_grad_(True)
    params.append(param)
for param in model.fc.parameters():
    param.requires_grad_(True)
    params.append(param)

model = model.to(device)
optimizer = optim.Adam(params, lr=lr)
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
        data = pool_threads.map(loadSequence, video_list)
        
        next_batch = 0
        for video in data: # For each video in video_list (i.e. the current batch)
            if video.size == 0: # If there is an empty video, skip the whole batch
                next_batch = 1
        if next_batch:
            continue
        
        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x), requires_grad=False).to(device).contiguous()
        #x = torch.FloatTensor(x).to(device).contiguous()
        y = train[1][random_indices[i:i+batch_size]]
        y = torch.from_numpy(y).to(device)
        
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4[0](h)
        h = model.avgpool(h)
        h = h.view(h.size(0), -1)
        output = model.fc(h)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = float(prediction.eq(y.data).sum()) / float(batch_size) * 100.0
        train_acc.append(accuracy)
        #print("Finish one batch!")
    
    avg_train_acc = np.mean(train_acc)
    epoch_train_acc.append(avg_train_acc)
    print("Epoch {}/{}, cumulative train accuracy: {}%".format(epoch+1, num_of_epochs, avg_train_acc))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    ## TEST
    model.eval()
    test_acc = []
    random_indices2 = np.random.permutation(len(test[0]))
    for i in range(0, len(test[0])-batch_size, batch_size):
        augment = False
        video_list2 = [(test[0][k], augment) for k in random_indices2[i:i+batch_size]]
        data2 = pool_threads.map(loadSequence, video_list2)
        
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
        
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            h = model.avgpool(h)
            h = h.view(h.size(0), -1)
            output = model.fc(h)
        
        prediction = output.data.max(1)[1]
        accuracy = float(prediction.eq(y.data).sum()) / float(batch_size) * 100.0
        test_acc.append(accuracy)
    
    avg_test_acc = np.mean(test_acc)
    epoch_test_acc.append(avg_test_acc)
    print("Epoch {}/{}, cumulative test accuracy: {}%".format(epoch+1, num_of_epochs, avg_test_acc))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    is_best = (avg_test_acc == np.max(epoch_test_acc))
    save_checkpoint(model, is_best, "Sequenceframes_Video_AR.ckpt")
    dic = {"Epoch": epoch, "Train_Accuracy": epoch_train_acc, "Test_Accuracy": epoch_test_acc}
    torch.save(dic, "Sequenceframes_Video_AR.checkpoint.pth.tar")

pool_threads.close()
pool_threads.terminate()
