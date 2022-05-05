#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:51:11 2022

@author: chrisw
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pretrain import CompressiveSensingPretraining
from data import GenotypeDataModule
from modules import FCEncoder
import wandb
import yaml
import sys
from shutil import copyfile
import os
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # wandb args
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_project', default='genotype_supervised', type=str)
    parser.add_argument('--wandb_entity', default='chrisxx', type=str)
    parser.add_argument('--wandb_pretrained', default=None, type=str)
    parser.add_argument('--checkpoint_yaml', default='ls_val_loss_checkpoint_callback.yaml')
    # datamodule args
    parser.add_argument('--path_pattern', default="datasets/genotype/cas9/cas9_pairs_10nm_%s.csv", type=str)
    parser.add_argument('--path', default=None)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    # fc args
    parser.add_argument('--d_model', default=8196//2, type=int)
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--d_hidden', type=int, default=8196//2)
    parser.add_argument('--embedding_size', type=int, default=20)
    # head args
    parser.add_argument('--l2_coef', default=1e-4, type=float)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)    
    # ------------
    # data
    # ------------
    if args.path is not None:
        paths = [args.path]
    else:
        pattern = args.path_pattern
        paths = [pattern%'train', pattern%'valid', pattern%'test']
    datamodule = GenotypeDataModule(batch_size=args.batch_size, 
                                         num_workers=args.num_workers,
                                         seed=args.seed,
                                         paths=paths)

    # ------------
    # model
    # ------------
    datamodule.prepare_data()
    datamodule.setup()
    n_feats = datamodule.get_n_feats()
    
    if args.wandb_pretrained is None:
        encoder = FCEncoder(16, args.embedding_size, args.d_model, n_feats, d_hidden = args.d_hidden, num_hidden_layers=args.num_hidden_layers)
    else:
        
        run = wandb.init(mode="online",
                 project='genotype_pretraining', 
                 entity='chrisxx', 
                 job_type="inference",
                 dir=".",
                 settings=wandb.Settings(start_method='fork'))
        model_at = run.use_artifact("%s:latest"%args.wandb_pretrained)
        model_dir = model_at.download(root='./artifacts/%s/'%args.wandb_pretrained)
        with open(model_dir+"/config.yaml") as file:
            pconfig = yaml.load(file, Loader=yaml.FullLoader)
        with open(model_dir+"/%s"%args.checkpoint_yaml) as file:
            scores = yaml.load(file, Loader=yaml.FullLoader)
        
        run.finish()
        
        mkey = None
        mscore = np.inf
        for key, loss in scores.items():
            if loss < mscore:
                mscore = loss
                mkey = key
        print('using %s'%mkey)
        encoder = FCEncoder(16, pconfig['embedding_size'], pconfig['d_model'], 23, pconfig['d_hidden'], pconfig['num_hidden_layers'])
        pmodel = CompressiveSensingPretraining.load_from_checkpoint('./artifacts/%s/%s'%(args.wandb_pretrained, mkey.split('/')[-1]), encoder=encoder)
        encoder = pmodel
    
    # ------------
    # wandb 
    # ------------
    wandb_logger = WandbLogger(entity=args.wandb_entity, 
                               project=args.wandb_project, 
                               name=args.wandb_name,
                               config=args)
    run = wandb_logger.experiment
        
    # ------------
    # create features
    # ------------
    def featurize(loader, encoder = encoder):
        #encoder.eval()
        X = []
        Y = []
        for x, y in loader:
            X += [encoder(x).detach().cpu().numpy()]
            Y += [y.detach().cpu().numpy()]
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        Y = Y[:, 0]
        return X, Y
    
    X_train, Y_train = featurize(datamodule.train_dataloader())
    X_valid, Y_valid = featurize(datamodule.val_dataloader())
    X_test, Y_test = featurize(datamodule.test_dataloader())
    
    #X_train = (X_train - X_train.mean())/X_train.std()
    #print(np.linalg.norm(X_train[:X_train.shape[0]].T.dot(X_train[:X_train.shape[0]])/(X_train.shape[0]) - np.eye(X_train.shape[1]))**2)
    #plt.matshow(X_train[:X_train.shape[0]].T.dot(X_train[:X_train.shape[0]])/(X_train.shape[0]))
    #plt.show()
    
    # train
    w_opt = np.linalg.solve(np.dot(X_train.T, X_train) + args.l2_coef*np.eye(X_train.shape[1]), np.dot(X_train.T, Y_train))    
    pred_train = X_train.dot(w_opt)
    pred_test = X_test.dot(w_opt)
    # validate
    pred_valid = X_valid.dot(w_opt)
    # test
    pred_test = X_test.dot(w_opt)
    run.log({'train_total_r2': r2_score(Y_train, pred_train),
             'valid_total_r2': r2_score(Y_valid, pred_valid), 
             'test_total_r2': r2_score(Y_test, pred_test)})
    
    


if __name__ == '__main__':
    main()
