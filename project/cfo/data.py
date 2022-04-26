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
                 paths=['datasets/cfo/suite/bitcount/no_reps_bitcount_1_LLVM_10_1000.csv'],
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
            self.n_flags = self.X.shape[1]
        elif len(self.paths) == 3:
            self.X_train, self.y_train = self._csv_to_tensors(self.paths[0])
            self.X_valid, self.y_valid = self._csv_to_tensors(self.paths[1])
            self.X_test, self.y_test = self._csv_to_tensors(self.paths[2]) 
            self.n_flags = self.X_train.shape[1]
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
    
    def get_n_flags(self):
        self.prepare_data()
        return self.n_flags
    
    def _csv_to_tensors(self, path):
        df = pd.read_csv(path)
        df = df.replace([np.nan], -1)
        data = df.to_numpy()
        sequences = torch.tensor(data[:, 1:], dtype=torch.long) + 1
        labels = data[:,0]
        labels = (labels - labels.mean())/labels.std()
        labels = torch.tensor(labels[:, np.newaxis], dtype=torch.float32)
        return sequences, labels
    

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
    def __init__(self, n_options, length=10000, no_augmentations=False, only_neighbors=False):
        self.n_options = n_options
        self.gen = UniformOptimizations(n_options)
        self.length=length
        self.no_augmentations = no_augmentations
        self.only_neighbors = only_neighbors
        
    def __getitem__(self, idx):
        plus_padding = np.zeros(self.n_options, dtype=np.int64)
        plus_padding2 = np.zeros(self.n_options, dtype=np.int64)
        next_opt = self.gen.randPerm()
        i = np.random.randint(len(next_opt))
        j = np.random.randint(len(next_opt))
        next_opt2 = next_opt.copy()
        if not self.no_augmentations:
            if self.only_neighbors:
                if i+1 < len(next_opt):
                    next_opt2[i] = next_opt[i+1]
                    next_opt2[i+1] = next_opt[i]
                elif i-1 >= 0:
                    next_opt2[i] = next_opt[i-1]
                    next_opt2[i-1] = next_opt[i]
            else:
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
                 only_neighbors=False,
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
        self.only_neighbors = only_neighbors
        self.save_hyperparameters()
        
    def prepare_data(self):
        self.downstream_dm.prepare_data()
        
    def setup(self, stage=None):
        self.downstream_dm.setup(stage)
        
    def train_dataloader(self):
        #print('creating new dataset...')
        dataset = AugmentedOptimizationsDataset(self.n_flags, self.n_train, no_augmentations=self.no_augmentations, 
                                                only_neighbors=self.only_neighbors)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def test_dataloader(self):
        val = self.downstream_dm.val_dataloader()
        return val
    
    def get_n_flags(self):
        return self.n_flags
