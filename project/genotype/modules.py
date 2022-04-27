#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:41:15 2022

@author: chrisw
"""

from torch import nn

class FCEncoder(nn.Module):
    def __init__(self, dict_size, embedding_size, d_model, max_len, d_hidden=4000):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embedding_size)
        self.fc = nn.Linear(embedding_size * max_len, d_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(d_hidden, d_model)
        self.flatten = nn.Flatten()
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.d_model = d_model
    
    def forward(self, x):
        return self.head(self.relu(self.fc(self.flatten(self.embed(x)))))
    
    def get_output_size(self):
        return self.d_model
        


