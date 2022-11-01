#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:41:15 2022

@author: chrisw
"""

from torch import nn

class FCEncoder(nn.Module):
    def __init__(self, dict_size, embedding_size, d_model, max_len, d_hidden=8192, num_hidden_layers=2):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embedding_size)
        self.fc = nn.Linear(embedding_size * max_len, d_hidden)
        self.bn = nn.BatchNorm1d(d_hidden)
        if num_hidden_layers > 0:
            self.hidden = nn.Sequential(*([nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True)]*num_hidden_layers))
        else:
            self.hidden = None
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(d_hidden, d_model)
        self.flatten = nn.Flatten()
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.d_model = d_model
    
    def forward(self, x):
        h = self.relu(self.bn(self.fc(self.flatten(self.embed(x)))))
        if self.hidden is not None:
            h = self.hidden(h)
        return self.head(h)
    
    def get_output_size(self):
        return self.d_model
        

class FCEncoderLegacy(nn.Module):
    def __init__(self, dict_size, embedding_size, d_model, max_len, d_hidden=4000, num_hidden_layers=0):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embedding_size)
        self.fc = nn.Linear(embedding_size * max_len, d_hidden)
        if num_hidden_layers > 0:
            self.hidden = nn.Sequential(*([nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True)]*num_hidden_layers))
        else:
            self.hidden = None
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(d_hidden, d_model)
        self.flatten = nn.Flatten()
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.d_model = d_model
    
    def forward(self, x):
        h = self.relu(self.fc(self.flatten(self.embed(x))))
        if self.hidden is not None:
            h = self.hidden(h)
        return self.head(h)
    
    def get_output_size(self):
        return self.d_model
        


