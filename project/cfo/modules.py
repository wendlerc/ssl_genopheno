#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:41:15 2022

@author: chrisw
"""

import torch
from torch import nn
import pytorch_lightning as pl
from einops import rearrange
import math


class FCEncoder(nn.Module):
    def __init__(self, dict_size, embedding_size, d_model, max_len, d_hidden=4000):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embedding_size)
        self.fc = nn.Linear(embedding_size * max_len, d_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(d_hidden, d_model)
        self.flatten = nn.Flatten()
        #self.relu = nn.ReLU(inplace=True) # for some reason works way better without relu
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.d_model = d_model
    
    def forward(self, x):
        return self.head(self.relu(self.fc(self.flatten(self.embed(x)))))
    
    def get_output_size(self):
        return self.d_model
        


class PositionalEncodingBatchFirst(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1) # maxlen x 1
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class LearntPositionalEncodingBatchFirst(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model), requires_grad=True)
        nn.init.normal_(self.pe, mean=.0, std=.5)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LSTMEncoder(nn.Module):
    def __init__(self, dict_size, 
                 embedding_size, 
                 hidden_size, 
                 num_layers=1, 
                 bias=True, 
                 batch_first=True, 
                 dropout=0, 
                 bidirectional=False,
                 proj_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(dict_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, 
                           num_layers=num_layers,
                           bias=bias,
                           batch_first=batch_first,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           proj_size=proj_size)
        
    def forward(self, x):
        embedded_sequence = self.embed(x)
        output, (hn, cn) = self.rnn(embedded_sequence)
        return torch.concat((output[:, -1], output.sum(dim=1)), dim=1)
    
    def get_output_size(self):
        return 2*self.hidden_size
    

class TransformerEncoder(nn.Module):
    def __init__(self, dict_size, max_len,
                 padding_token=0,
                 num_layers=1, 
                 norm=None,
                 d_model=256, 
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5,
                 batch_first=True, 
                 norm_first=False):
        super().__init__()
        self.padding_token = padding_token
        self.d_model = d_model
        self.embed = nn.Embedding(dict_size, d_model)
        self.pos_enc = PositionalEncodingBatchFirst(d_model, max_len=max_len, dropout=dropout)
        self.enc = nn.Transformer(d_model=d_model, 
                                  nhead=nhead,
                                  num_encoder_layers=num_layers,
                                  num_decoder_layers=num_layers,
                                  dim_feedforward=dim_feedforward,
                                  dropout=dropout, 
                                  activation=activation,
                                  batch_first=batch_first,
                                  norm_first=norm_first)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        
        
        
    def forward(self, x):
        # x has shape n_batch x n_seq (n_seq = max_len for now)
        src_key_padding_mask = (x == self.padding_token) # n_batch x n_seq
        
        batch = self.embed(x)
        batch = self.pos_enc(math.sqrt(self.d_model) * batch)
        
        out = self.enc(batch, self.cls_token.expand(x.shape[0], -1, -1), src_key_padding_mask=src_key_padding_mask)
        return out[:,0,:]
    
    def get_output_size(self):
        return self.d_model



class SelfAttentionEncoder(nn.Module):
    def __init__(self, dict_size, max_len,
                 padding_token=0,
                 num_layers=1, 
                 norm=None,
                 d_model=256, 
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5,
                 batch_first=True, 
                 norm_first=False):
        super().__init__()
        self.padding_token = padding_token
        self.d_model = d_model
        self.embed = nn.Embedding(dict_size, d_model)
        self.pos_enc = LearntPositionalEncodingBatchFirst(d_model, max_len=max_len, dropout=dropout)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout, 
                                                                    activation=activation,
                                                                    layer_norm_eps=layer_norm_eps,
                                                                    batch_first=batch_first,
                                                                    norm_first=norm_first), num_layers, norm=norm)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        
        
        
    def forward(self, x):
        # x has shape n_batch x n_seq (n_seq = max_len for now)
        src_key_padding_mask = (x == self.padding_token) # n_batch x n_seq
        
        batch = self.embed(x)
        batch = self.pos_enc(math.sqrt(self.d_model) * batch)
        
        
        src_key_padding_mask = torch.cat((torch.zeros((batch.shape[0], self.cls_token.shape[0]), 
                                                      dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x 1+n_seq
        
        self_attn = self.enc(torch.cat((self.cls_token.expand(x.shape[0], -1, -1), batch), dim=1),
                             src_key_padding_mask=src_key_padding_mask)

        return self_attn[:, 0] # only return embeddings of the cls tokens
    
    def get_output_size(self):
        return self.d_model