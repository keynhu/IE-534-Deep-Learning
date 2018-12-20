# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy
import os
import shutil
import re

import torch
import torch.nn as nn
import torch.utils.data as Data

import torchvision
import torchvision.transforms as transforms

from PIL import Image

# Create validation set divided with labels
# Load validation labels
file2 = open("tiny-imagenet-200/val/val_annotations.txt", "r")
valpic = []
vallabel = []
tt = file2.readlines()
for line in tt:
    lines = re.split("\t",line)
    valpic.append(lines[0])
    vallabel.append(lines[1])
file2.close()
# Create directories and move pictures
valfolder = "tiny-imagenet-200/val/images/"
alllabel = list(set(vallabel))
for label in alllabel:
    os.makedirs(valfolder+label)
for (pic,label) in zip(valpic,vallabel):
    shutil.move(valfolder+pic, valfolder+label)

# In validation set, transform all grey images to "RGB", all image sizes to (224 x 224 x 3), and save (No Use)
start_time = datetime.datetime.now()
print("Start Time of validation transform: {}".format(start_time))
valfolder = "tiny-imagenet-200/val/images/"
valall = os.listdir(valfolder)
trans_rz = transforms.Resize(size=(224,224))
for i, vfold in enumerate(valall):
    valcls = os.listdir(os.path.join(valfolder,vfold))
    for image in valcls:
        picpath = os.path.join(valfolder, vfold, image)
        pic = Image.open(picpath)
        if pic.mode == "L":
            pic = pic.convert("RGB")
        pic = trans_rz(pic)
        pic.save(picpath,"JPEG") # Then save the new pic with the same name
    if (i+1)%20==0:
        print("{} classes of pictures in validation set have been transformed...".format(i+1))
        now_time = datetime.datetime.now()
        print("Cost time: {}".format(now_time-start_time))

start_time = datetime.datetime.now()
print("Start Time of training transform: {}".format(start_time))
trainfolder = "tiny-imagenet-200/train"
trainall = os.listdir(trainfolder)
trans_rz = transforms.Resize(size=(224,224))
for j, tfold in enumerate(trainall):
    traincls = os.listdir(os.path.join(trainfolder, tfold, "Images"))
    for image in traincls:
        picpath = os.path.join(trainfolder, tfold, "Images", image)
        pic = Image.open(picpath)
        if pic.mode == "L":
            pic = pic.convert("RGB")
        pic = trans_rz(pic)
        pic.save(picpath, "JPEG") # Then save the new pic with the same name
    if (j+1)%10==0:
        print("{} classes of pictures in training set have been transformed...".format(j+1))
        now_time = datetime.datetime.now()
        print("Cost time: {}".format(now_time-start_time))