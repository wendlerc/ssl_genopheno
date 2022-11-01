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


letter2int = {'A':1, 'C':2, 'G':3, 'T':4}
int2letter = {1:'A', 2:'C', 3:'G', 4:'T'}
pair2int = {('A', 'A'): 0, ('A', 'C'): 1, ('A', 'G'): 2, ('A', 'T'): 3, ('C', 'A'): 4, ('C', 'C'): 5, ('C', 'G'): 6, ('C', 'T'): 7, ('G', 'A'): 8, ('G', 'C'): 9, ('G', 'G'): 10, ('G', 'T'): 11, ('T', 'A'): 12, ('T', 'C'): 13, ('T', 'G'): 14, ('T', 'T'): 15}
int2pair = {val:key for key, val in pair2int.items()}

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
                 select_subset=False,
                 random=False,
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
        self.select_subset = select_subset
        self.random = random
        self.save_hyperparameters()
        
    def prepare_data(self):
        if len(self.paths) == 1:
            self.X, self.y = self._csv_to_tensors(self.paths[0])
            self.n_feats = self.X.shape[1]
        elif len(self.paths) == 3:
            self.X_train, self.y_train = self._csv_to_tensors(self.paths[0])
            if self.select_subset:
                n_train = int(len(self.X_train)*self.frac_train)
                if self.random:
                    perm = np.random.permutation(len(self.X_train))
                    self.X_train = self.X_train[perm[:n_train]]
                    self.y_train = self.y_train[perm[:n_train]]
                else:
                    self.X_train = self.X_train[:n_train]
                    self.y_train = self.y_train[:n_train]
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
    def __init__(self, gene_string, n_feats, length=10000, no_augmentations=False, hard=False, easy=False, genotype_list=None, pairs=True, n_augs=2, mix=False):
        self.n_feats = n_feats
        self.length=length
        self.no_augmentations = no_augmentations
        self.hard = hard
        self.easy = easy
        self.genotype_list = genotype_list
        self.gene_string = gene_string
        self.impossible_mutations = np.asarray([letter2int[base] for base in gene_string])
        self.pairs = pairs
        self.n_augs = n_augs
        self.mix = mix
        #print(self.impossible_mutations)
        #print(self.impossible_mutations.shape)
        if genotype_list is not None:
            self.length = len(genotype_list)
    
    
    def augment_(self, d0, idcs, n_augs):
        d1 = d0.copy()
        idcs = np.where((d0 != 0)*(d0 != 5)*(d0 != 10)*(d0 != 15))[0]
        for aug in range(min(len(idcs), n_augs)):
            idx1 = idcs[np.random.randint(len(idcs))]
            new_base1 = int2letter[np.random.randint(1, 5)]
            while pair2int[(self.gene_string[idx1], new_base1)] in [pair2int[(self.gene_string[idx1], self.gene_string[idx1])], d0[idx1]]:
                new_base1 = int2letter[np.random.randint(1, 5)]
            d1[idx1] = pair2int[(self.gene_string[idx1], new_base1)]
        if self.hard:
            idcs_nomutations = np.where(~((d0 != 0)*(d0 != 5)*(d0 != 10)*(d0 != 15)))[0]
            if np.random.rand() > 0.9:
                # -> figure out which mutations are plausible and do the right one: 
                idx = np.random.choice(idcs_nomutations)
                pair = int2pair[d1[idx]]
                d1[idx] = pair2int[(pair[0], int2letter[np.random.randint(1, 5)])]
        return d1
                
    
    
    def __getitem__(self, idx):
        if self.genotype_list is None:
            d0 = np.random.randint(0, 5, self.n_feats)
        else:
            d0 = self.genotype_list[idx]

        if self.pairs:
            if self.mix:
                idcs = np.where((d0 != 0)*(d0 != 5)*(d0 != 10)*(d0 != 15))[0]
                n_augs = np.random.randint(0, min(len(idcs)+1, self.n_augs+1), 2)
                d1 = self.augment_(d0, idcs, n_augs[0])
                d2 = self.augment_(d0, idcs, n_augs[1])
            else: # legacy  
                d1 = d0.copy()
                d2 = d0.copy()
                idcs = np.where((d0 != 0)*(d0 != 5)*(d0 != 10)*(d0 != 15))[0]
                for aug in range(min(len(idcs), self.n_augs)):
                    idx1 = idcs[np.random.randint(len(idcs))]
                    idx2 = idcs[np.random.randint(len(idcs))]
                    new_base1 = int2letter[np.random.randint(1, 5)]
                    new_base2 = int2letter[np.random.randint(1, 5)]
                    while pair2int[(self.gene_string[idx1], new_base1)] in [pair2int[(self.gene_string[idx1], self.gene_string[idx1])], d0[idx1]]:
                        new_base1 = int2letter[np.random.randint(1, 5)]
                    while pair2int[(self.gene_string[idx2], new_base2)] in [pair2int[(self.gene_string[idx2], self.gene_string[idx2])], d0[idx2]]:
                        new_base2 = int2letter[np.random.randint(1, 5)]
                    d1[idx1] = pair2int[(self.gene_string[idx1], new_base1)]
                    d2[idx2] = pair2int[(self.gene_string[idx2], new_base2)]
                
        else:
            d1 = d0.copy()
            if not self.no_augmentations:
                new = np.random.randint(1, 5, self.n_feats)
                if self.easy:
                    idcs = np.where(d1 > 0)[0]
                    if len(idcs) > 0: 
                        idx = idcs[np.random.randint(len(idcs))]
                        d2[idx] = 0
                elif not self.hard:
                    idcs = np.where(d1 > 0)[0]
                    if len(idcs) > 0:
                        idx = idcs[np.random.randint(len(idcs))]
                        new_base = np.random.randint(1, 5)
                        while new_base == self.impossible_mutations[idx]:
                            new_base = np.random.randint(1, 5)
                        d2[idx] = new_base
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
                 gene_string,
                 batch_size=512, 
                 num_workers=12,
                 n_train=100000,
                 no_augmentations=False,
                 only_neighbors=False,
                 hard=False,
                 genotype_list = None,
                 n_augs = 2,
                 mix = False,
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
        self.genotype_list = genotype_list
        self.gene_string = gene_string
        self.n_augs = n_augs
        self.mix = mix
        self.save_hyperparameters()
        
        
    def prepare_data(self):
        self.downstream_dm.prepare_data()
        
    def setup(self, stage=None):
        self.downstream_dm.setup(stage)
        
    def train_dataloader(self):
        #print('creating new dataset...')
        dataset = AugmentedGenotypeDataset(self.gene_string, self.n_feats, self.n_train, no_augmentations=self.no_augmentations, hard=self.hard, 
                                           genotype_list = self.genotype_list, n_augs=self.n_augs, mix=self.mix)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        val = self.downstream_dm.train_dataloader()
        return val
    
    def test_dataloader(self):
        val = self.downstream_dm.train_dataloader()
        return val
    
    def get_n_feats(self):
        return self.n_feats
