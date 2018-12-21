# Import modules
import numpy as np
import itertools
import os
import shutil
import io

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()
        self.lstm = nn.LSTMCell(in_size, out_size)
        
        self.out_size = out_size
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None
    
    def forward(self,x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.h = torch.zeros(state_size).to(device)
            self.c = torch.zeros(state_size).to(device)
        self.h, self.c = self.lstm(x, (self.h, self.c))
        return self.h
    
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x
    
class RNN_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units) #,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        # self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        # self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        #self.loss = nn.CrossEntropyLoss() # When doing multiple classification
        self.loss = nn.BCEWithLogitsLoss() # When doing binary classification

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        # self.lstm2.reset_state()
        # self.dropout2.reset_state()

    def forward(self, x, t, train=True):

        embed = self.embedding(x) # (batch_size, time_steps, features(i.e. no_of_hidden_units))

        no_of_timesteps = embed.shape[1] # Length of sentence

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train) # (batch_size, no_of_hidden_units)

            # h = self.lstm2(h)
            # h = self.bn_lstm2(h)
            # h = self.dropout2(h,dropout=0.3,train=train)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps, batch_size, features)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps), representing a batch of sentences

        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)
        h = h.view(h.size(0),-1) # (batch_size, features)
        #h = self.dropout(h)

        h = self.fc_output(h) # (batch_size, 1)

        return self.loss(h[:,0],t), h[:,0] #F.softmax(h, dim=1)    