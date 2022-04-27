#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:03:26 2022

@author: chrisw
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import pandas as pd
import numpy as np
import torch


class GenotypeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size=512, 
                 num_workers=12,
                 frac_train=0.8,
                 frac_val=0.1,
                 seed=42,
                 paths=['datasets/genotype/cas9/cas9_train.csv',
                        'datasets/genotype/cas9/cas9_valid.csv',
                        'datasets/genotype/cas9/cas9_test.csv'],
                 *args,
                 **kwargs):
        """
        Note that 0 is used as padding token, 1,...,99 are the tokens for the flags
        
        paths: list of paths, if it only has one element, random splits are performed, else
        the provided files are interpreted as train, valid, test
        """
        super().__init__()
        self.paths = paths
        if type(self.paths) is str:
            self.paths = [self.paths]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.seed = seed
        self.save_hyperparameters()
        
    def prepare_data(self):
        if len(self.paths) == 1:
            self.X, self.y = self._csv_to_tensors(self.paths[0])
            self.n_feats = self.X.shape[1]
        elif len(self.paths) == 3:
            self.X_train, self.y_train = self._csv_to_tensors(self.paths[0])
            self.X_valid, self.y_valid = self._csv_to_tensors(self.paths[1])
            self.X_test, self.y_test = self._csv_to_tensors(self.paths[2]) 
            self.n_feats = self.X_train.shape[1]
        else:
            raise NotImplementedError('please provide either one file or three files.')
        
    def setup(self, stage=None):
        if len(self.paths) == 1:
            n = len(self.X)
            dataset = TensorDataset(self.X, self.y)
            self.train, self.valid, self.test = random_split(dataset, [int(self.frac_train*n), 
                                                                       int(self.frac_val*n), 
                                                                       n - int(self.frac_train*n) - int(self.frac_val*n)], generator=torch.Generator().manual_seed(self.seed))
        elif len(self.paths) == 3:
            self.train = TensorDataset(self.X_train, self.y_train)
            self.valid = TensorDataset(self.X_valid, self.y_valid)
            self.test = TensorDataset(self.X_test, self.y_test)
        else:
            raise NotImplementedError('please provide either one file or three files.')
            
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def get_n_feats(self):
        self.prepare_data()
        return self.n_feats
    
    def _csv_to_tensors(self, path):
        df = pd.read_csv(path)
        data = df.to_numpy()
        X = torch.tensor(data[:, 1:-1], dtype=torch.long) 
        y = torch.tensor(data[:,-1][:,np.newaxis], dtype=torch.float32)
        return X, y
    
        
class AugmentedGenotypeDataset(Dataset):
    def __init__(self, n_feats, length=10000, no_augmentations=False, hard=False):
        self.n_feats = n_feats
        self.length=length
        self.no_augmentations = no_augmentations
        self.hard = hard
        
    def __getitem__(self, idx):
        d1 = np.random.randint(0, 5, self.n_feats)
        d2 = d1.copy()
        if not self.no_augmentations:
            new = np.random.randint(1, 5, self.n_feats)
            if not self.hard:
                idcs = np.where(d1 > 0)[0]
                idx = idcs[np.random.randint(len(idcs))]
                d2[idx] = new[idx]
            else:
                d2[d1 > 0] = new[d1 > 0]
        if self.no_augmentations:
            d2 = d1.copy()

        tensor1 = torch.tensor(d1, dtype=torch.long)
        tensor2 = torch.tensor(d2, dtype=torch.long)
        return tensor1, tensor2
    
    def __len__(self):
        return self.length


class AugmentedGenotypePretrainingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 downstream_datamodule,
                 batch_size=512, 
                 num_workers=12,
                 n_train=100000,
                 no_augmentations=False,
                 only_neighbors=False,
                 hard=False,
                 *args,
                 **kwargs):
        """
        Note that 0 is used as padding token, 1,...,64 are the tokens for the flags
        """
        super().__init__()
        self.n_train = n_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downstream_dm = downstream_datamodule
        self.n_feats = downstream_datamodule.get_n_feats()
        self.no_augmentations = no_augmentations
        self.only_neighbors = only_neighbors
        self.hard = hard
        self.save_hyperparameters()
        
        
    def prepare_data(self):
        self.downstream_dm.prepare_data()
        
    def setup(self, stage=None):
        self.downstream_dm.setup(stage)
        
    def train_dataloader(self):
        #print('creating new dataset...')
        dataset = AugmentedGenotypeDataset(self.n_feats, self.n_train, no_augmentations=self.no_augmentations, hard=self.hard)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def test_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def get_n_feats(self):
        return self.n_feats
