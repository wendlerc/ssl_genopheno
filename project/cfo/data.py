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
from utils import UniformOptimizations


class OptimizationsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size=512, 
                 num_workers=12,
                 frac_train=0.8,
                 frac_val=0.1,
                 seed=42,
                 #path='datasets/cfo/suite/bitcount/bitcount_1_LLVM_99_1000.csv',
                 path='datasets/cfo/suite/bitcount/no_reps_bitcount_1_LLVM_10_1000.csv',
                 #path='datasets/cfo/suite/bitcount/perm_bitcount_1_LLVM_99_1000.csv',
                 *args,
                 **kwargs):
        """
        Note that 0 is used as padding token, 1,...,99 are the tokens for the flags
        """
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.seed = seed
        self.save_hyperparameters()
        
    def prepare_data(self):
        df = pd.read_csv(self.path)
        df = df.replace([np.nan], -1)
        data = df.to_numpy()
        self.n_flags = data.shape[1] - 1
        self.sequences = torch.tensor(data[:, 1:], dtype=torch.long) + 1
        labels = data[:,0]
        labels = (labels - labels.mean())/labels.std()
        self.labels = torch.tensor(labels[:, np.newaxis], dtype=torch.float32)
        
    def setup(self, stage=None):
        n = len(self.sequences)
        dataset = TensorDataset(self.sequences, self.labels)
        self.train, self.valid, self.test = random_split(dataset, [int(self.frac_train*n), 
                                                                   int(self.frac_val*n), 
                                                                   n - int(self.frac_train*n) - int(self.frac_val*n)], generator=torch.Generator().manual_seed(self.seed))
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def get_n_flags(self):
        self.prepare_data()
        return self.n_flags
    

class UniformOptimizationsDataset(Dataset):
    def __init__(self, n_options, length=10000):
        self.n_options = n_options
        self.gen = UniformOptimizations(n_options)
        self.length=length
        
    def __getitem__(self, idx):
        plus_padding = np.zeros(self.n_options, dtype=np.int64)
        next_opt = self.gen.randPerm()
        plus_padding[:len(next_opt)] = next_opt
        return torch.tensor(plus_padding, dtype=torch.long)
    
    def __len__(self):
        return self.length


class OptimizationsPretrainingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 downstream_datamodule,
                 batch_size=512, 
                 num_workers=12,
                 n_train=10000,
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
        self.n_flags = downstream_datamodule.get_n_flags()
        self.save_hyperparameters()
        
    def prepare_data(self):
        self.downstream_dm.prepare_data()
        
    def setup(self, stage=None):
        self.downstream_dm.setup(stage)
        
    def train_dataloader(self):
        #print('creating new dataset...')
        dataset = UniformOptimizationsDataset(self.n_flags, self.n_train)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def test_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val

    def get_n_flags(self):
        return self.n_flags
    
### version with augmentations
        
class AugmentedOptimizationsDataset(Dataset):
    def __init__(self, n_options, length=10000, no_augmentations=False):
        self.n_options = n_options
        self.gen = UniformOptimizations(n_options)
        self.length=length
        self.no_augmentations = no_augmentations
        
    def __getitem__(self, idx):
        plus_padding = np.zeros(self.n_options, dtype=np.int64)
        plus_padding2 = np.zeros(self.n_options, dtype=np.int64)
        next_opt = self.gen.randPerm()
        i = np.random.randint(len(next_opt))
        j = np.random.randint(len(next_opt))
        next_opt2 = next_opt.copy()
        if not self.no_augmentations:
            next_opt2[i] = next_opt[j]
            next_opt2[j] = next_opt[i]
        plus_padding[:len(next_opt)] = next_opt
        plus_padding2[:len(next_opt2)] = next_opt2
        tensor1 = torch.tensor(plus_padding, dtype=torch.long)
        tensor2 = torch.tensor(plus_padding2, dtype=torch.long)
        return tensor1, tensor2
    
    def __len__(self):
        return self.length


class AugmentedOptimizationsPretrainingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 downstream_datamodule,
                 batch_size=512, 
                 num_workers=12,
                 n_train=100000,
                 no_augmentations=False,
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
        self.n_flags = downstream_datamodule.get_n_flags()
        self.no_augmentations = no_augmentations
        self.save_hyperparameters()
        
    def prepare_data(self):
        self.downstream_dm.prepare_data()
        
    def setup(self, stage=None):
        self.downstream_dm.setup(stage)
        
    def train_dataloader(self):
        #print('creating new dataset...')
        dataset = AugmentedOptimizationsDataset(self.n_flags, self.n_train, no_augmentations=self.no_augmentations)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def test_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def get_n_flags(self):
        return self.n_flags
