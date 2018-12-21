# Import modules
import numpy as np
import itertools
import nltk
import os
import shutil
import io

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.autograd.variable as Variable

import torchvision
import torchvision.transforms as transforms

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
    
class RNN_language_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(RNN_language_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units) #,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.lstm2 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm2 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)
        
        self.lstm3 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm3 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout3 = LockedDropout() #torch.nn.Dropout(p=0.5)        

        self.decoder = nn.Linear(no_of_hidden_units, vocab_size) # Output a probability distribution over all words
        self.vocab_size = vocab_size

        self.loss = nn.CrossEntropyLoss()#ignore_index=0) # When doing multiple classification

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        self.lstm2.reset_state()
        self.dropout2.reset_state()
        self.lstm3.reset_state()
        self.dropout3.reset_state()

    def forward(self, x, t, train=True):

        embed = self.embedding(x) # (batch_size, time_steps, features(i.e. no_of_hidden_units))

        no_of_timesteps = embed.shape[1] # Length of sentence

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.3,train=train) # (batch_size, no_of_hidden_units)

            h = self.lstm2(h)
            h = self.bn_lstm2(h)
            h = self.dropout2(h,dropout=0.3,train=train)

            h = self.lstm3(h)
            h = self.bn_lstm3(h)
            h = self.dropout3(h,dropout=0.3,train=train)
            
            h = self.decoder(h)
            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps, batch_size, vocab_size)
        target_pred = outputs.permute(1,0,2) # (batch_size, time_steps, vocab_size)
        outputs = outputs.permute(1,2,0) # (batch_size, vocab_size, time_steps), representing a batch of sentences

        if train:
            target_pred = target_pred.contiguous().view(-1, self.vocab_size)
            target = x[:,1:].contiguous.view(-1)
            loss = self.loss(target_pred, target)
            return loss, outputs

        return outputs