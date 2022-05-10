#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:41:15 2022

@author: chrisw
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv
import math

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
        self.pos_enc = PositionalEncodingBatchFirst(d_model, max_len=max_len, dropout=dropout)
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
        #
        src_key_padding_mask = torch.cat((torch.zeros((batch.shape[0], self.cls_token.shape[0]), 
                                                      dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x 1+n_seq
        self_attn = self.enc(torch.cat((self.cls_token.expand(x.shape[0], -1, -1), batch), dim=1),
                             src_key_padding_mask=src_key_padding_mask)
        
        return self_attn[:, 0] # only return embeddings of the cls tokens
    
    def get_output_size(self):
        return self.d_model


############# Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self, num_node_feats, emb_dim, num_classes, embedding=False, path_to_embedding=None):
        super().__init__()
        self.conv1 = GCNConv(num_node_feats, emb_dim) if not embedding else torch.load(path_to_embedding)
        self.conv2 = GCNConv(emb_dim, num_classes)
    
        self.relu = nn.ReLU()
        self.softmax =  nn.Softmax(dim=-1)
        self.dropout = nn.Dropout()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        #deleted softmax from here
        return x

class SparseClassifier(torch.nn.Module):
    def __init__(self, embedding_layer, emb_dim, num_classes) -> None:
        super().__init__()
        self.net1 = embedding_layer
        self.linear_classifier = nn.Linear(emb_dim, num_classes, bias=False)

    def forward(self, data):
        x = self.net1(data)
        x = self.linear_classifier(x)
        return x