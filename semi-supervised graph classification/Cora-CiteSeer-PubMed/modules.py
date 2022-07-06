#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:41:15 2022

@author: chrisw
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAT, GIN


############# Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self, num_node_feats, emb_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_feats, emb_dim) 
        self.conv2 = GCNConv(emb_dim, num_classes)
    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        #deleted softmax from here
        return x

class customGAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GAT(in_channels=in_dim, hidden_channels=2 * out_dim, num_layers=2, out_channels=out_dim) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat(x, edge_index)
        return x

class customGIN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gin = GIN(in_channels=in_dim, hidden_channels=2 * out_dim, num_layers=2, out_channels=out_dim) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin(x, edge_index)
        return x

class GBT_GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, batch_normalization=False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 2 * out_dim) 
        self.conv2 = GCNConv(2 * out_dim, out_dim)
    
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(2 * out_dim, momentum=0.01)
        self.bn = batch_normalization
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        if(self.bn):
            x = self.bn1(x)
        x = self.prelu(x)
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