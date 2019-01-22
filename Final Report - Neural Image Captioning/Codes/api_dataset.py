import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
import skimage.io as io
from random import seed, choice, sample
from scipy.misc import imread, imresize
from pycocotools.coco import COCO
class CaptionDataset(Dataset):
    def __init__(self, image_folder, caption_path, word_map, split, max_len_caption=100, captions_per_image=5, transform=None):   
        self.split = split    
        capFile = caption_path
        coco_caps = COCO(capFile)
        imgIds = coco_caps.getImgIds()
        self.imgs = list()
        # Captions per image
        self.cpi = captions_per_image
        # Load encoded captions and caption lengths (completely into memory)
        self.captions = list()
        self.caplens = list()        
        for i in range(len(imgIds)):
            img = coco_caps.loadImgs(imgIds[i])[0]
            filename = image_folder + img['file_name']
            I = imread(filename)
            if len(I.shape) == 2: # deal with images with only grey scale
                I = I[:, :, np.newaxis]
                I = np.concatenate([I, I, I], axis=2)
            I = imresize(I, (256, 256))
            I = I.transpose(2, 0, 1)
            assert I.shape == (3, 256, 256)
            assert np.max(I) <= 255
            self.imgs.append(I)
            annIds = coco_caps.getAnnIds(imgIds=img['id']) 
            anns = coco_caps.loadAnns(annIds) #A list with 5 dictionaries           
            enc_captions = list()
            caplens = list()
            tokens = [caption['caption'].replace('.', '').split() for caption in anns]

            for token in tokens:
                if len(token) > max_len_caption:
                    tokens.remove(token)
            if len(tokens) < captions_per_image:
                tokens = tokens + [choice(tokens) for _ in range(captions_per_image - len(tokens))]
            else:
                tokens = sample(tokens, k=captions_per_image)
            enc_captions = list()
            caplens = list()
            for j, c in enumerate(tokens):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len_caption - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                enc_captions.append(enc_c)
                caplens.append(c_len)
            self.captions.extend(enc_captions)
            self.caplens.extend(caplens)
        self.transform = transform
        self.dataset_size = len(self.captions)           

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])                    
        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions
    def __len__(self):
        return self.dataset_size
