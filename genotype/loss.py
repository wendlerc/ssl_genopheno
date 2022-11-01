#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:00:36 2022

@author: chrisw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class CompressiveSensingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor):
        # cross-correlation matrix
        c = batch.T @ batch / batch.shape[0] # DxD
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + off_diag
        #return 1/(batch.shape[1]**2)*(on_diag + off_diag)
        
class BarlowTwinsLoss(nn.Module):
    def __init__(self, C=1.):
        super().__init__()
        self.C = C

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor):
        # cross-correlation matrix
        c = batch1.T @ batch2 / batch1.shape[0] # DxD
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.C * off_diag

class VicRegLoss(nn.Module):
    def __init__(self, repr_coef = 25., std_coef = 25., cov_coef = 1.): 
        # these are the hyperparams used for imagenet with a three layer expander network
        # with 8192 neurons each...
        super().__init__()
        self.repr_coef = repr_coef
        self.std_coef = std_coef 
        self.cov_coef = cov_coef
        
    def forward(self, out_1: torch.Tensor, out_2: torch.Tensor):
        N = out_1.size(0)
        D = out_1.size(1)
        x = out_1
        y = out_2
        
        # s(Z, Z'), "unit": 1/(N*D)
        repr_loss = F.mse_loss(x, y) 
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        # v(Z), "unit": compute std's of the columns and then average the columnns -> 1/(N*D)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        
        # C(Z), "unit": the DxD dot products are normalized by the batch size, then in VicReg reference
        # implementation only divided by D, not by D**2 as mean() would do .... try the VicReg one first...
        cov_x = (x.T @ x) / (N - 1)
        cov_y = (y.T @ y) / (N - 1)
        # this would have the same "unit" as the other stuff
        # cov_loss = off_diagonal(cov_x).pow_(2).mean()/2 + \
        #           off_diagonal(cov_y).pow_(2).mean()/2
        
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) + \
                   off_diagonal(cov_y).pow_(2).sum().div(D)
        
        loss = self.repr_coef * repr_loss \
             + self.std_coef * std_loss \
             + self.cov_coef * cov_loss
        return loss