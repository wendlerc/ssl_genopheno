#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:03:26 2022

@author: chrisw
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import torch


class OptimizationsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 n_flags=64,
                 batch_size=512, 
                 num_workers=12,
                 seed=42,
                 path='datasets/cfo/bitcount_1_GCC_64_1000.csv',
                 *args,
                 **kwargs):
        """
        Note that 0 is used as padding token, 1,...,64 are the tokens for the flags
        """
        super().__init__()
        self.path = path
        self.n_flags = n_flags
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.save_hyperparameters()
        
    def prepare_data(self):
        df = pd.read_csv(self.path)
        df = df.replace([np.nan], -1)
        data = df.to_numpy()
        self.sequences = torch.tensor(data[:, 1:], dtype=torch.long) + 1
        labels = data[:,0]
        labels = (labels - labels.mean())/labels.std()
        self.labels = torch.tensor(labels[:, np.newaxis], dtype=torch.float32)
        
    def setup(self, stage=None):
        n = len(self.sequences)
        dataset = TensorDataset(self.sequences, self.labels)
        self.train, self.valid, self.test = random_split(dataset, [int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)], generator=torch.Generator().manual_seed(self.seed))
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
        