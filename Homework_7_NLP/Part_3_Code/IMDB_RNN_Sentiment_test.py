# Import modules
import numpy as np
import itertools
import datetime
import sys
import os
import shutil
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

from util_Sentiment import RNN_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load test data
vocab_size = 8000 # Set this because the 8000 most common words include over 95% of all the words
x_test = []
with io.open('HW7/preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

# Hyperparameters
vocab_size += 1 # Save 0 as unknown
embedding_size = 500
batch_size = 200
num_epochs = 10
L_Y_test = len(y_test)

# Build model
model = torch.load("rnn_sentiment.ckpt")

start_time = datetime.datetime.now()
print("Training starts at: ", start_time)

# Test process
test_acc = []
for epoch in range(num_epochs):
    epoch_test_cor = 0
    epoch_test_loss = 0
    epoch_test_all = 0
    model.eval()
    I_permut2 = np.random.permutation(L_Y_test)
    sequence_length = (epoch+1)*50

    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_test[j] for j in I_permut2[i:i+batch_size]]
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_test[I_permut2[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).to(device)
        target = Variable(torch.FloatTensor(y_input)).to(device)
        
        with torch.no_grad():
            loss, output = model(data, target, train=False)
        
        pred = (output >= 0)
        truth = (target>=0.5)
        epoch_test_cor += pred.eq(truth).sum().cpu().data.numpy()
        epoch_test_loss += loss.data.item()
        epoch_test_all += batch_size
        
    period_acc = epoch_test_cor / epoch_test_all
    period_loss = epoch_test_loss / (epoch_test_all / batch_size)
    test_acc.append(period_acc)
       
    print("Sequence Length: {}, cumulative test accuracy: {}%, test loss: {}".format(sequence_length, period_acc*100, period_loss))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
