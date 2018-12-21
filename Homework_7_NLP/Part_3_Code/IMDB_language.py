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

from util_language import RNN_language_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Save function
def save_checkpoint(obj, is_best, filename="checkpoint.pth.tar"):
    torch.save(obj, filename)
    if is_best:
        shutil.copyfile(filename, "best_"+filename)

# Load train data
#imdb_dictionary = np.load('HW7/preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 # Set this because the 8000 most common words include over 95% of all the words
x_train = []
with io.open('HW7/preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

# Load test data
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
num_epochs = 75
lr = 1e-3
L_Y_train = len(y_train)
L_Y_test = len(y_test)

# Build model
model = RNN_language_model(vocab_size, embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

start_time = datetime.datetime.now()
print("Training starts at: ", start_time)

# Train and test process
train_loss = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    # this avoids overflow
    if epoch > 5:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / 10
    
    # TRAIN
    epoch_cor = 0
    epoch_loss = 0
    epoch_all = 0
    model.train() # Change to train mode
    I_permut = np.random.permutation(L_Y_train)
    
    for i in range(0, L_Y_train, batch_size):
        x_input2 = [x_train[j] for j in I_permut[i:i+batch_size]]
        sequence_length = 50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        data = Variable(torch.LongTensor(x_input)).to(device)

        optimizer.zero_grad()
        loss, output = model(data, train=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        values, pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy()
        accuracy = np.sum(pred==data.cpu().data.numpy()[:,1:]) / sequence_length # Total accurate count in batch
        epoch_cor += accuracy
        epoch_loss += loss.data.item()
        epoch_all += batch_size
    
    period_acc = epoch_cor / epoch_all
    period_loss = epoch_loss / (epoch_all / batch_size)
    train_acc.append(period_acc)
    train_loss.append(period_loss)
        
    print("Epoch {}/{}, cumulative train accuracy: {}%, train loss: {}".format(epoch+1, num_epochs, period_acc*100, period_loss))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
    
    # TEST
    epoch_test_cor = 0
    epoch_test_all = 0
    model.eval()
    I_permut2 = np.random.permutation(L_Y_test)
    
    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_test[j] for j in I_permut2[i:i+batch_size]]
        sequence_length = 100
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        data = Variable(torch.LongTensor(x_input)).to(device)
        
        with torch.no_grad():
            output = model(data, train=False)
        
        values, pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy()
        accuracy = np.sum(pred==data.cpu().data.numpy()[:,1:]) / sequence_length
        epoch_test_cor += accuracy
        epoch_test_all += batch_size
        
    period_acc = epoch_test_cor / epoch_test_all
    test_acc.append(period_acc)
    
    is_best = (period_acc == np.max(test_acc))
    save_checkpoint(model, is_best, "language_rnn.ckpt")
    data = {"Epoch":num_epochs, "Train_loss": train_loss, "Train_acc": train_acc, "Test_acc": test_acc}
    save_checkpoint(data, is_best, 'language_rnn_checkpoint.pth.tar')
    
    print("Epoch {}/{}, cumulative test accuracy: {}%".format(epoch+1, num_epochs, period_acc*100))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
