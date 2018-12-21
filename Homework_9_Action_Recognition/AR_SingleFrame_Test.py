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

import h5py
try:
    import cv2
except ImportError:
    print("Trying to Install required module: cv2\n")
    os.system("python -m pip install --user opencv-python")
    import cv2

from multiprocessing import Pool
from util_AR import save_checkpoint, getUCF101, loadFrame

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMAGE_SIZE = 224
NUM_CLASSES = 101
frame_batch_size = 100
lr = 1e-4

# Load data
data_dir = "/projects/training/bauh/AR/"
class_list, _, test = getUCF101(data_dir)
print(len(test[0]))

# Introduce pretrained ResNet-50 model
model = torch.load("best_Singleframe_Video_AR.ckpt")
model = model.to(device)

# Save prediction directory
#pred_dir = 'HW9/Part_1/Output/'
#for label in class_list:
#    if not os.path.exists(pred_dir+label+"/"):
#        os.mkdir(pred_dir+label+"/")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad_(False)

# Activate multiple CPU cores
# pool_threads = Pool(8, maxtasksperchild=200)

acc_top1 = 0
acc_top5 = 0
acc_top10 = 0
predictions = []
prob_dist = np.zeros((len(test[0]), NUM_CLASSES), dtype=np.float32)
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
full_labels = test[1][random_indices]
mean = np.asarray([0.485, 0.456, 0.406], np.float32)
std = np.asarray([0.229, 0.224, 0.225], np.float32)
model.eval()

start_time = datetime.datetime.now()
print("Start training at: ", start_time)

for i in range(len(test[0])):
    index = random_indices[i]
    filename = test[0][index]
    filename = filename.replace(".avi", ".hdf5")
    filename = filename.replace("UCF-101", "UCF-101-hdf5")

    h = h5py.File(filename, 'r')
    nFrames = len(h["video"])

    data = np.zeros((nFrames, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frame = (frame - mean) / std
        frame = frame.transpose(2,0,1)
        data[j,:,:,:] = frame
    h.close()

    pred = np.zeros((nFrames, NUM_CLASSES), dtype=np.float32)
    loop_i = list(range(0, nFrames, frame_batch_size))
    loop_i.append(nFrames)

    for j in range(len(loop_i)-1):
        data_batch = data[loop_i[j]:loop_i[j+1]]

        with torch.no_grad():
            x = np.asarray(data_batch, dtype=np.float32)
            x = torch.FloatTensor(x).to(device).contiguous()

            output = model(x)
        pred[loop_i[j]:loop_i[j+1]] = output.cpu().numpy()

    #filename = filename.replace(data_dir+"UCF-101-hdf5", pred_dir)
    # Softmax
    for j in range(pred.shape[0]):
        pred[j] = np.exp(pred[j]) / np.sum(np.exp(pred[j]))
    pred = np.sum(np.log(pred), axis=0) # Add up all predictions on each frame in this video
    prob_dist[i,:] = pred
    argsort_pred = np.argsort(-pred)[0:10]
    
    label = test[1][index]
    predictions.append(argsort_pred[0])
    confusion_matrix[label, argsort_pred[0]] += 1
    if label == argsort_pred[0]:
        acc_top1 += 1
    if label in argsort_pred[0:5]:
        acc_top5 += 1
    if label in argsort_pred:
        acc_top10 += 1
    
    if (i+1)%100 == 0:
        now_time = datetime.datetime.now()
        print("Test images {}/{}, Accuracy: ({}, {}, {})".format(i+1, len(test[0]), acc_top1/(i+1), acc_top5/(i+1), acc_top10/(i+1)))
        print("Cost time: ", now_time-start_time)
    
num_of_pred = np.sum(confusion_matrix, axis=1)
for j in range(NUM_CLASSES):
    confusion_matrix[j,:] = confusion_matrix[j,:] / np.sum(confusion_matrix[j,:])
        
num_truepos = np.diag(confusion_matrix)
order_truepos = np.argsort(num_truepos)
    
sorted_list = np.asarray(class_list)
sorted_list = sorted_list[order_truepos]
sorted_truepos = num_truepos[order_truepos]
    
output = pd.DataFrame({"Classes": sorted_list, "Classes_true": sorted_truepos, "Classes_positive": num_of_pred[order_truepos],
"ACC_1": acc_top1/len(test[0]), "ACC_5": acc_top5/len(test[0]), "ACC_10":acc_top10/len(test[0])})
output.to_csv("HW9/Part_1/Output/SingleFrame_test.csv")
full_pred = pd.DataFrame({"Labels": full_labels, "Predictions": predictions})
full_pred.to_csv("HW9/Part_1/Output/SingleFrame_predictions.csv")
np.save('HW9/Part_1/Output/single_frame_confusion_matrix.npy', confusion_matrix)
np.save('HW9/Part_1/Output/single_frame_prob_distribution.npy', prob_dist)
