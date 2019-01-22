# Import modules
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import datetime
import json

# Set up directories
data_folder = "/projects/training/bauh/COCO/"
train_img_folder = data_folder + "train2014"
val_img_folder = data_folder + "val2014"
train_annot = data_folder + "annotations/captions_train2014.json"
val_annot = data_folder + "annotations/captions_val2014.json"

# Read in the annotation json files
cap_train_ins = open(train_annot).read()
cap_val_ins = open(val_annot).read()
cap_train = json.loads(cap_train_ins)
cap_val = json.loads(cap_val_ins)

# Build a dictionary object
data_dic = {"images": [], "dataset": "coco"}

start_time = datetime.datetime.now()
print("Start Time: ", start_time)

# Save filenames with image ids
img_train_dic = {}
for image in cap_train["images"]:
    tmp_id = image["id"]
    tmp_name = image["file_name"]
    img_train_dic.update({tmp_id: tmp_name})

now_time = datetime.datetime.now()
print("Train filenames saved!")
print("Cost Time: ", now_time-start_time)

img_val_dic = {}
for image in cap_val["images"]:
    tmp_id = image["id"]
    tmp_name = image["file_name"]
    img_val_dic.update({tmp_id: tmp_name})

now_time = datetime.datetime.now()
print("Validation filenames saved!")
print("Cost Time: ", now_time-start_time)
    
# Iterate over all train annotations
data_dic_train_img_id = []
for i, annot in enumerate(cap_train["annotations"]):
    tmp_caption = annot["caption"].replace('.','')
    tmp_img_id = annot["image_id"]
    tmp_sent_id = annot["id"]
    
    if not tmp_img_id in data_dic_train_img_id:
        data_dic["images"].append({"cocoid":tmp_img_id, "filename":img_train_dic[tmp_img_id], "filepath":train_img_folder,
                                  "sentences":[{"raw": tmp_caption, "sent_id":tmp_sent_id, "tokens":tmp_caption.split()}],
                                  "sent_ids":[tmp_sent_id], "split":"train"})
        data_dic_train_img_id.append(tmp_img_id)
    else:
        tmp_index = data_dic_train_img_id.index(tmp_img_id)
        tmp_dic = data_dic["images"][tmp_index]
        data_dic["images"][tmp_index]["sentences"].append({"raw": tmp_caption, "sent_id":tmp_sent_id, "tokens":tmp_caption.split()})
        data_dic["images"][tmp_index]["sent_ids"].append(tmp_sent_id)
    if (i+1)%1000 == 0:
        now_time = datetime.datetime.now()
        print("{} annotations finished, cost Time: {}".format(i+1, now_time-start_time))

# Build a json file with data_dic
with open("final/sorted_coco_train.json", "w") as outfile:
    json.dump(data_dic, outfile)

now_time = datetime.datetime.now()
print("Train dictionary saved!")
print("Cost Time: ", now_time-start_time)

# Iterate over all valid annotations
data_dic_val_img_id = []
for j, annot in enumerate(cap_val["annotations"]):
    tmp_caption = annot["caption"].replace('.','')
    tmp_img_id = annot["image_id"]
    tmp_sent_id = annot["id"]
    
    if not tmp_img_id in data_dic_val_img_id:
        data_dic["images"].append({"cocoid":tmp_img_id, "filename":img_val_dic[tmp_img_id], "filepath":val_img_folder,
                                  "sentences":[{"raw": tmp_caption, "sent_id":tmp_sent_id, "tokens":tmp_caption.split()}],
                                  "sent_ids":[tmp_sent_id], "split":"val"})
        data_dic_val_img_id.append(tmp_img_id)
    else:
        tmp_index = data_dic_val_img_id.index(tmp_img_id)
        tmp_dic = data_dic["images"][tmp_index]
        data_dic["images"][tmp_index]["sentences"].append({"raw": tmp_caption, "sent_id":tmp_sent_id, "tokens":tmp_caption.split()})
        data_dic["images"][tmp_index]["sent_ids"].append(tmp_sent_id)
    if (j+1)%1000 == 0:
        now_time = datetime.datetime.now()
        print("{} annotations finished, cost Time: {}".format(j+1, now_time-start_time))

# Build a json file with data_dic
with open("final/sorted_coco.json", "w") as outfile:
    json.dump(data_dic, outfile)    

now_time = datetime.datetime.now()
print("Valid dictionary saved!")
print("Cost Time: ", now_time-start_time)    
