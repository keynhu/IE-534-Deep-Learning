import numpy as np
import h5py
import time
import os
import io
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

from util_language import RNN_language_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
temperature = 0.5 # float(sys.argv[1])
length_of_review = 100

# Count time
start_time = datetime.datetime.now()
print("Training starts at: ", start_time)

# Load dictionary
imdb_dictionary = np.load('HW7/preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1
word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}
# Load model
model = torch.load("best_language_rnn.ckpt").to(device)
model.eval()
print("Dictionary loaded!")
now_time = datetime.datetime.now()
print("Cost time: ", now_time - start_time)

# Load tokens
## all tokens should be the same length if doing more than one
## tokens = [['i','love','this','movie','.'],['i','hate','this','movie','.']]
tokens = [['i'],['a']] # All reviews will begin with these two words (or partial sentences)
token_ids = np.asarray([[word_to_id.get(token,-1)+1 for token in x] for x in tokens]) # num_of_review, time_steps

## preload phrase
x = Variable(torch.LongTensor(token_ids)).to(device)
embed = model.embedding(x) # num_of_review, time_steps, features
state_size = [embed.shape[0],embed.shape[2]] # num_of_review, features
no_of_timesteps = embed.shape[1]

model.reset_state()

outputs = []
for i in range(no_of_timesteps):
    h = model.lstm1(embed[:,i,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    h = model.decoder(h)

    outputs.append(h)

outputs = torch.stack(outputs)
outputs = outputs.permute(1,2,0)
output = outputs[:,:,-1] # num_of_review, vocab_size -- log probability of next word
print("Review initialized!")
now_time = datetime.datetime.now()
print("Cost time: ", now_time - start_time)

review = []
for j in range(length_of_review):
    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1)
    review.append(x.cpu().data.numpy()[:,0])

    ## predict the next word
    embed = model.embedding(x)

    h = model.lstm1(embed[:,i,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    output = model.decoder(h)
print("Review generated!")
now_time = datetime.datetime.now()
print("Cost time: ", now_time - start_time)

# Print reviews
txt_name = "Review_Temp_{}.txt".format(temperature)
review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)
    with open(txt_name, "a") as f:
        f.writelines(prnt_str + '\n')
