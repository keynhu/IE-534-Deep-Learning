from utils import create_input_files
import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from collections import Counter
from random import seed, choice, sample
create_input_files(dataset='coco',
                    karpathy_json_path='final/data/dataset_coco.json',
                    image_folder='/projects/training/bauh/COCO',
                    captions_per_image=5,
                    min_word_freq=5,
                    output_folder='scratch',
                    max_len=50)
