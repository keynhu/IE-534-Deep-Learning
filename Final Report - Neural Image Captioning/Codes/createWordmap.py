import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from collections import Counter
from random import seed, choice, sample

def create_word_map(json_path, min_word_freq, output_folder, dataset_name='coco'):
    # Read the json file that stores captions
    with open(json_path, 'r') as j:
        data = json.load(j)
    
    # Count word frequencies
    word_freq = Counter()
    for img in data['images']: # One image
        for c in img['sentences']: # Several captions
            # Update word frequency
            word_freq.update(c['tokens'])
            
    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0  # word_map: {'<pad>':0, 'a':1, 'b':2, 'c':3, '<unk>':4, '<start>':5, '<end>':6}
    
    # Create a base/root name for all output files
    base_filename = dataset_name + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
