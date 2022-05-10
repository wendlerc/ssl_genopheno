#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:00:36 2022

@author: chrisw
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class CompressiveSensingLoss(nn.Module):
    def __init__(self, lamda=1):
        super().__init__()
        self.lamda = lamda
        
    def forward(self, batch: torch.Tensor):
        # cross-correlation matrix
        # batch = (batch - batch.mean(0)) / batch.std(0) 
        c = batch.T @ batch / batch.shape[0] # DxD
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag +  self.lamda * off_diag 

def power_iteration(batch, num_simulations = 1):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    I = Variable(torch.eye(batch.shape[1]), requires_grad=True).to(device)
    A = batch.T @ batch - I
    b_k = torch.rand(A.shape[0]).to(device)
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k

        # re normalize the vector
        b_k = b_k1 / torch.linalg.norm(b_k1)

    return b_k.T @ A @ b_k / (torch.linalg.norm(b_k) ** 2)
    # return torch.linalg.norm(A @ b_k) ** 2

EPS = 1e-15

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )
