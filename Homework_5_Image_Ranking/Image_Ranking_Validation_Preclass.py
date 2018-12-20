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