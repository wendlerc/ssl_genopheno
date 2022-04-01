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

from data import OptimizationsPretrainingDataModule, OptimizationsDataModule
from modules import SelfAttentionEncoder
from loss import CompressiveSensingLoss
import wandb
import yaml


class CompressiveSensingPretraining(pl.LightningModule):
    def __init__(self, encoder,
                 lr = 1e-3,
                 beta1 = 0.9,
                 beta2 = 0.95,
                 factor = 0.5):
        super().__init__()
        self.encoder = encoder       
        self.bn = nn.BatchNorm1d(encoder.get_output_size(), affine=False)# this makes sure that dropout does not mess up our loss
        self.my_lr_arg = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.factor = factor
        self.loss = CompressiveSensingLoss()
        
    def forward(self, x):
        return self.bn(self.encoder(x))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.my_lr_arg, betas=(self.beta1, self.beta2))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.factor)
        return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':'val_loss'}
    
    def training_step(self, batch, batch_idx):
        x = batch
        pred = self.forward(x)
        loss = self.loss(pred)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred, y
    
    def validation_epoch_end(self, outputs):
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
        Y_test_mean = Y_test.mean()
        r2 = np.sum((Y_pred - Y_test_mean)**2)/np.sum((Y_test - Y_test_mean)**2)
        loss = np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test)
        self.log('downstream R2', r2)
        self.log('val_loss', loss)
        print('downstream R2 %2.4f loss %2.4f'%(r2, loss))
        return r2
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # wandb args
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_project', default='cfo_compressive_sensing_pretraining', type=str)
    parser.add_argument('--wandb_entity', default='chrisxx', type=str)
    # datamodule args
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    # lightingmodule args
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--factor', default=0.5, type=float)
    # selfattn args
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    # trainer args
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_monitor', type=str, default='val_loss')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=2)
    parser.add_argument('--early_stopping_monitor', type=str, default='val_loss')
    parser.add_argument('--early_stopping_mode', type=str, default='min')
    parser.add_argument('--early_stopping_patience', type=int, default=25)
    parser.add_argument('--my_log_every_n_steps', type=int, default=1)
    parser.add_argument('--my_accelerator', type=str, default='gpu')
    parser.add_argument('--my_max_epochs', type=int, default=5)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    # ------------
    # wandb 
    # ------------
    wandb_logger = WandbLogger(entity=args.wandb_entity, 
                               project=args.wandb_project, 
                               name=args.wandb_name,
                               config=args)
    
    # ------------
    # data
    # ------------
    ddm = OptimizationsDataModule(batch_size=args.batch_size, 
                                         num_workers=args.num_workers,
                                         frac_train=0.0,
                                         frac_val=1.0,
                                         seed=args.seed)
    datamodule = OptimizationsPretrainingDataModule(ddm, batch_size=args.batch_size, 
                                         num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    encoder = SelfAttentionEncoder(65, 64, 
                                   num_layers=args.num_layers,
                                   d_model=args.d_model,
                                   nhead=args.nhead,
                                   dim_feedforward=args.dim_feedforward,
                                   dropout=args.dropout,
                                   activation=args.activation,
                                   layer_norm_eps=args.layer_norm_eps)
    model = CompressiveSensingPretraining(encoder, lr=args.lr, 
                               beta1=args.beta1, 
                               beta2=args.beta2,
                               factor=args.factor)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                          save_top_k=args.checkpoint_save_top_k,
                                          monitor=args.checkpoint_monitor)
    es_callback = EarlyStopping(monitor=args.early_stopping_monitor, 
                                mode=args.early_stopping_mode, 
                                patience=args.early_stopping_patience)
    lr_monitor = LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback,
                                                       es_callback, lr_monitor],
                                            log_every_n_steps=args.my_log_every_n_steps,
                                            accelerator=args.my_accelerator,
                                            max_epochs=args.my_max_epochs,
                                            reload_dataloaders_every_n_epochs=1)#this is important for pretraining with dataloaders that just generate examples
    trainer.fit(model, datamodule=datamodule)
    # ------------
    # valdiation
    # -----------
    trainer.validate(datamodule=datamodule, ckpt_path='best')
    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=datamodule, ckpt_path='best')
    print(result)
   
    print("uploading model...")
    #store config and model
    run = wandb_logger.experiment
    checkpoint_callback.to_yaml(checkpoint_callback.dirpath+'/checkpoint_callback.yaml')
    with open(checkpoint_callback.dirpath+'/config.yaml', 'w') as f:
        yaml.dump(run.config.as_dict(), f, default_flow_style=False)
    
    trained_model_artifact = wandb.Artifact(run.name, type="model", description="trained selfattn model")
    trained_model_artifact.add_dir(checkpoint_callback.dirpath)
    run.log_artifact(trained_model_artifact)


if __name__ == '__main__':
    main()
