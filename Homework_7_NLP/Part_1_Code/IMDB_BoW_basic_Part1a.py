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

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# Build model class
class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, t):
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).to(device)
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],t), h[:,0]

# Hyperparameters
vocab_size += 1 # Save 0 as unknown
embedding_size = 500
batch_size = 200
num_epochs = 20
lr = 1e-1
L_Y_train = len(y_train)
L_Y_test = len(y_test)

# Build model
model = BOW_model(vocab_size, embedding_size).to(device)
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Train process
train_loss = []
train_acc = []
test_acc = []

start_time = datetime.datetime.now()
print("Training starts at: ", start_time)

for epoch in range(num_epochs):
    # this avoids overflow
    if epoch > 20:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
    epoch_cor = 0
    epoch_loss = 0
    epoch_all = 0
    model.train() # Change to train mode
    I_permut = np.random.permutation(L_Y_train)
    
    for i in range(0, L_Y_train, batch_size):
        x_input = [x_train[j] for j in I_permut[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permut[i:i+batch_size]], dtype=np.int)
        target = torch.FloatTensor(y_input).to(device)
        
        optimizer.zero_grad()
        loss, output = model(x_input, target)
        loss.backward()
        optimizer.step()
        
        pred = (output >= 0)
        truth = (target>=0.5)
        epoch_cor += pred.eq(truth).sum().cpu().data.numpy()
        epoch_loss += loss.data.item()
        epoch_all += batch_size
        
    period_acc = epoch_cor / epoch_all
    period_loss = epoch_loss / (epoch_all / batch_size)
    train_acc.append(period_acc)
    train_loss.append(period_loss)
        
    print("Epoch {}/{}, cumulative train accuracy: {}%, train loss: {}".format(epoch+1, num_epochs, period_acc*100, period_loss))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)
        
    # Test
    epoch_test_cor = 0
    epoch_test_loss = 0
    epoch_test_all = 0
    model.eval()
    I_permut2 = np.random.permutation(L_Y_test)
    
    for i in range(0, L_Y_test, batch_size):
        x_input = [x_test[j] for j in I_permut2[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permut2[i:i+batch_size]], dtype=np.int)
        target = torch.FloatTensor(y_input).to(device)
        
        with torch.no_grad():
            loss, output = model(x_input, target)
        
        pred = (output >= 0)
        truth = (target>=0.5)
        epoch_cor += pred.eq(truth).sum().cpu().data.numpy()
        epoch_loss += loss.data.item()
        epoch_all += batch_size
        
    period_acc = epoch_cor / epoch_all
    period_loss = epoch_loss / (epoch_all / batch_size)
    test_acc.append(period_acc)
       
    print("Epoch {}/{}, cumulative test accuracy: {}%, test loss: {}".format(epoch+1, num_epochs, period_acc*100, period_loss))
    now_time = datetime.datetime.now()
    print("Cost Time: ", now_time-start_time)

torch.save(model, "BoW_basic.ckpt")
data = {"Epoch":num_epochs, "Train_loss": train_loss, "Train_acc": train_acc, "Test_acc": test_acc, "Batch_size": batch_size, "Optimizer": "SGD", "Learning_rate": lr, "Embedding_size":embedding_size}
torch.save(data, 'BoW_basic_checkpoint.pth.tar')
