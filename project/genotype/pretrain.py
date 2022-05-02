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
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import AugmentedGenotypePretrainingDataModule, GenotypeDataModule
from modules import FCEncoder
from loss import CompressiveSensingLoss, off_diagonal, BarlowTwinsLoss
import wandb
import yaml

from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, LassoLarsCV

import glob
import sys
from shutil import copyfile
import os
import pandas as pd


class CompressiveSensingPretraining(pl.LightningModule):
    def __init__(self, encoder,
                 lr = 1e-3,
                 beta1 = 0.9,
                 beta2 = 0.95,
                 factor = 0.5,
                 monitor = 'mean_train_loss'):
        super().__init__()
        self.encoder = encoder       
        self.bn = nn.BatchNorm1d(encoder.get_output_size(), affine=False)# this makes sure that dropout does not mess up our loss
        self.my_lr_arg = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.factor = factor
        self.loss = BarlowTwinsLoss()
        self.monitor = monitor
        
    def forward(self, x):
        return self.bn(self.encoder(x))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.my_lr_arg, betas=(self.beta1, self.beta2), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.factor)
        return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor': self.monitor}
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        pred1 = self.forward(x1)
        pred2 = self.forward(x2)
        loss = self.loss(pred1, pred2)
        self.log('train_loss', loss)
        return {'loss': loss, 'z1': pred1, 'z2': pred2}
    
    def training_epoch_end(self, outputs):
        z1 = outputs[-1]['z1']
        z2 = outputs[-1]['z2']
        c = z1.T @ z2 / z1.shape[0] # DxD
        wandb.log({'crosscorrelation': wandb.Image(c)})
        self.log('cc_off_diag_min', off_diagonal(c**2).min())
        self.log('cc_off_diag_max', off_diagonal(c**2).max())
        self.log('cc_off_diag_median', off_diagonal(c**2).median())
        print(c.shape)
        mean_loss = torch.mean(torch.stack([o['loss'] for o in outputs]))
        if hasattr(self.encoder, 'cls_token'):
            self.log('cls_token_norm', torch.sum(self.encoder.cls_token**2).item())
        self.log('mean_train_loss', mean_loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred, y
    
    
    def validation_epoch_end_lasso(self, outputs):
        X = []
        Y = []
        for batch in outputs:
            x, y = batch
            X += [x.detach().cpu().numpy()]
            Y += [y.detach().cpu().numpy()]
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        Y = Y[:, 0]        
        n_train = int(0.8 * len(X))
        X_train = X[:n_train]
        Y_train = Y[:n_train]
        X_test = X[n_train:]
        Y_test = Y[n_train:]
        #reg = LassoCV(cv=5, eps=1e-3, max_iter=10000)
        reg = LassoLarsCV(cv=5)
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        loss = np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test)
        self.log('lasso downstream R2', r2)
        self.log('lasso_val_loss', loss)
        print('Lasso: downstream R2 %2.4f loss %2.4f'%(r2, loss))
        return loss
    
    def validation_epoch_end_least_squares(self, outputs):
        X = []
        Y = []
        for batch in outputs:
            x, y = batch
            X += [x.detach().cpu().numpy()]
            Y += [y.detach().cpu().numpy()]
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        Y = Y[:, 0]        
        n_train = int(0.8 * len(X))
        X_train = X[:n_train]
        Y_train = Y[:n_train]
        X_test = X[n_train:]
        Y_test = Y[n_train:]
        w_opt = np.linalg.solve(np.dot(X_train.T, X_train), np.dot(X_train.T, Y_train))       
        Y_pred = X_test.dot(w_opt)
        #Y_test_mean = Y_test.mean()
        #r2 = 1 - np.sum((Y_pred - Y_test)**2)/np.sum((Y_test - Y_test_mean)**2)
        r2 = r2_score(Y_test, Y_pred)
        loss = np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test)
        self.log('ls downstream R2', r2)
        self.log('ls_val_loss', loss)
        print('LS: downstream R2 %2.4f loss %2.4f'%(r2, loss))
        return loss
        
    
    def validation_epoch_end(self, outputs):
        l1 = self.validation_epoch_end_least_squares(outputs)
        l2 = self.validation_epoch_end_lasso(outputs)
        self.log('val_loss', l2)
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def get_output_size(self):
        return self.encoder.get_output_size()
    

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # wandb args
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_project', default='genotype_pairs_compressive_sensing_pretraining', type=str)
    parser.add_argument('--wandb_entity', default='chrisxx', type=str)
    # datamodule args
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--no_augmentations', action='store_true')
    parser.add_argument('--only_neighbors', action='store_true')
    parser.add_argument('--path_pattern', type=str, default="datasets/genotype/cas9/cas9_pairs_10nm_%s.csv")
    parser.add_argument('--gene_string', type=str, default="GACGCATAAAGATGAGACGCTGG")
    parser.add_argument('--hard', action='store_true')
    # lightingmodule args
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--factor', default=0.5, type=float)
    # fc args
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--num_hidden_layers', default=0, type=int)
    parser.add_argument('--d_hidden', type=int, default=4000)
    parser.add_argument('--embedding_size', type=int, default=20)
    # trainer args
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    #parser.add_argument('--checkpoint_monitor', type=str, default='val_loss')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=2)
    #parser.add_argument('--early_stopping_monitor', type=str, default='val_loss')
    parser.add_argument('--early_stopping_mode', type=str, default='min')
    parser.add_argument('--early_stopping_patience', type=int, default=25)
    parser.add_argument('--monitor', type=str, default='mean_train_loss')
    parser.add_argument('--my_log_every_n_steps', type=int, default=1)
    parser.add_argument('--my_accelerator', type=str, default='gpu')
    parser.add_argument('--my_max_epochs', type=int, default=500)
    parser.add_argument('--n_augs', type=int, default=2)
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    # ------------
    # wandb 
    # ------------
    wandb_logger = WandbLogger(entity=args.wandb_entity, 
                               project=args.wandb_project, 
                               name=args.wandb_name,
                               config=args,
                               settings=wandb.Settings(start_method='fork'))
    
    run = wandb_logger.experiment
    # save file to artifact folder
    
    result_dir = args.checkpoint_dir+'/%s/'%wandb_logger.experiment.name 
    os.makedirs(result_dir, exist_ok=True)
    copyfile(sys.argv[0], result_dir+sys.argv[0].split('/')[-1])
    
    # ------------
    # data
    # ------------
    df = pd.read_csv(args.path_pattern%'full')
    genotype_list = df.to_numpy()[:, 1:-1]
    paths = [args.path_pattern%'train', args.path_pattern%'valid', args.path_pattern%'test']
    ddm = GenotypeDataModule(batch_size=args.batch_size, 
                                         num_workers=args.num_workers,
                                         frac_train=0.0,
                                         frac_val=1.0,
                                         seed=args.seed,
                                         paths=paths)
    datamodule = AugmentedGenotypePretrainingDataModule(ddm, args.gene_string, batch_size=args.batch_size, 
                                         num_workers=args.num_workers,
                                         n_train=args.n_train,
                                         no_augmentations=args.no_augmentations,
                                         only_neighbors=args.only_neighbors,
                                         hard = args.hard,
                                         genotype_list = genotype_list,
                                         n_augs = args.n_augs)
    #print(args.gene_string)
    #print(len(args.gene_string))
    #print('-------------------')
    #print(genotype_list.shape)
    #exit()


    # ------------
    # model
    # ------------
    n_feats = datamodule.get_n_feats()
    
    encoder = FCEncoder(16, args.embedding_size, args.d_model, n_feats, d_hidden=args.d_hidden, num_hidden_layers=args.num_hidden_layers)
        

    model = CompressiveSensingPretraining(encoder, lr=args.lr, 
                               beta1=args.beta1, 
                               beta2=args.beta2,
                               factor=args.factor,
                               monitor=args.monitor)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                          filename='{epoch}-{mean_train_loss:.2f}-{ls_val_loss:.2f}-{lasso_val_loss}-train',
                                          save_top_k=args.checkpoint_save_top_k,
                                          monitor=args.monitor)
    checkpoint_callback2 = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                           filename='{epoch}-{mean_train_loss:.2f}-{ls_val_loss:.2f}-{lasso_val_loss}-ls',
                                           save_top_k=args.checkpoint_save_top_k,
                                           monitor="ls_val_loss")
    checkpoint_callback3 = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                           filename='{epoch}-{mean_train_loss:.2f}-{ls_val_loss:.2f}-{lasso_val_loss}-lasso',
                                           save_top_k=args.checkpoint_save_top_k,
                                           monitor='lasso_val_loss')
    es_callback = EarlyStopping(monitor=args.monitor, 
                                mode=args.early_stopping_mode, 
                                patience=args.early_stopping_patience)
    lr_monitor = LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback,
                                                       checkpoint_callback2,
                                                       checkpoint_callback3,
                                                       es_callback, lr_monitor],
                                            log_every_n_steps=args.my_log_every_n_steps,
                                            accelerator=args.my_accelerator,
                                            max_epochs=args.my_max_epochs,
                                            reload_dataloaders_every_n_epochs=1)#this is important for pretraining with dataloaders that just generate examples
    
    # ------------
    # baseline valdiation
    # -----------
    trainer.validate(datamodule=datamodule, model=model)
    # ------------
    # baseline testing
    # ------------
    result = trainer.test(datamodule=datamodule, model=model)
    # ------------
    # training
    # ------------
    trainer.fit(model, datamodule=datamodule)
    # ------------
    # final valdiation
    # -----------
    trainer.validate(datamodule=datamodule, ckpt_path='best')
    # ------------
    # final testing
    # ------------
    result = trainer.test(datamodule=datamodule, ckpt_path='best')
    print(result)
   
    print("uploading model...")
    #store config and model
    checkpoint_callback.to_yaml(checkpoint_callback.dirpath+'/mean_train_loss_checkpoint_callback.yaml')
    checkpoint_callback2.to_yaml(checkpoint_callback.dirpath+'/ls_val_loss_checkpoint_callback.yaml')
    checkpoint_callback3.to_yaml(checkpoint_callback.dirpath+'/lasso_val_loss_checkpoint_callback.yaml')
    checkpoint_callback3.to_yaml(checkpoint_callback.dirpath+'/checkpoint_callback.yaml')
    
    with open(checkpoint_callback.dirpath+'/config.yaml', 'w') as f:
        yaml.dump(run.config.as_dict(), f, default_flow_style=False)
    
    trained_model_artifact = wandb.Artifact(run.name, type="model", description="trained selfattn model")
    trained_model_artifact.add_dir(checkpoint_callback.dirpath)
    run.log_artifact(trained_model_artifact)


if __name__ == '__main__':
    main()
