#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:00:36 2022

@author: chrisw
"""

import torch
import torch.nn as nn
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
