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

class GenotypeRegression(pl.LightningModule):
    def __init__(self, encoder,
                 lr = 1e-3,
                 beta1 = 0.9,
                 beta2 = 0.95,
                 factor = 0.5,
                 l1_coef = 1.,
                 freeze = False,
                 scores = {'r2': torchmetrics.R2Score},
                 monitor = 'mean_valid_loss'):
        super().__init__()
        self.encoder = encoder        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        hidden_size = encoder.get_output_size()
        #self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
        #                           nn.ReLU(inplace=True),
        #                           nn.Linear(hidden_size, hidden_size),
        #                           nn.ReLU(inplace=True),
        #                           nn.Linear(hidden_size, 1))
        self.head = nn.Linear(hidden_size, 1)
        self.l1_coef = l1_coef
        self.scores = scores
        self.train_scores = nn.ModuleDict({key: score() for key, score in scores.items()})
        self.valid_scores = nn.ModuleDict({key: score() for key, score in scores.items()})
        self.test_scores = nn.ModuleDict({key: score() for key, score in scores.items()})
        
        self.my_lr_arg = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.factor = factor
        self.monitor = monitor
        
    def forward(self, x):
        if self.freeze:
            self.encoder.eval()
        return self.head(self.encoder(x))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.my_lr_arg, betas=(self.beta1, self.beta2), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.factor)
        return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':self.monitor}
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        if self.l1_coef > 0:
            l1 = sum(p.abs().mean() for p in self.head.parameters())
            loss = F.mse_loss(pred, y) + self.l1_coef*l1
        else:
            loss = F.mse_loss(pred, y)
        self.log('train_loss', loss)
        for name, score in self.train_scores.items():
            self.log('train_%s'%name, score(pred, y))
        return loss
    
    def training_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.stack([o['loss'] for o in outputs]))
        if hasattr(self.encoder, 'cls_token'):
            self.log('cls_token_norm', torch.sum(self.encoder.cls_token**2).item())
        self.log('mean_train_loss', mean_loss)
        self.train_scores = nn.ModuleDict({key: score() for key, score in self.scores.items()}).to(self.device)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        self.log('val_loss', loss)
        for name, score in self.valid_scores.items():
            self.log('valid_%s'%name, score(pred, y))
        return loss
            
    def validation_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.stack(outputs))
        self.log('mean_valid_loss', mean_loss)
        for name, score in self.valid_scores.items():
            try:
                self.log('valid_total_%s'%name, score.compute())
            except ValueError:
                continue 
        self.valid_scores = nn.ModuleDict({key: score() for key, score in self.scores.items()}).to(self.device)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        self.log('test_loss', loss)
        for name, score in self.test_scores.items():
            self.log('test_%s'%name, score(pred, y))
        return loss
            
    def test_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.stack(outputs))
        self.log('mean_test_loss', mean_loss)
        for name, score in self.test_scores.items():
            try:
                self.log('test_total_%s'%name, score.compute())
            except ValueError:
                continue 
        self.test_scores = nn.ModuleDict({key: score() for key, score in self.scores.items()}).to(self.device)


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
    parser.add_argument('--checkpoint_yaml', default='checkpoint_callback.yaml')
    parser.add_argument('--artifact_dir', default="/cluster/scratch/wendlerc/artifacts", type=str)
    # datamodule args
    parser.add_argument('--path_pattern', default="datasets/genotype/cas9/cas9_pairs_10nm_%s.csv", type=str)
    parser.add_argument('--path', default=None)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--train_fraction', type=float, default=1.)
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--random', action='store_true')
    # lightingmodule args
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--l1_coef', default=0., type=float)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--group', default='default', type=str)
    # fc args
    parser.add_argument('--d_model', default=8196//2, type=int)
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--d_hidden', type=int, default=8196//2)
    parser.add_argument('--embedding_size', type=int, default=20)
    # trainer args
    parser.add_argument('--monitor', type=str, default='mean_valid_loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=2)
    parser.add_argument('--early_stopping_mode', type=str, default='min')
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    parser.add_argument('--my_log_every_n_steps', type=int, default=1)
    parser.add_argument('--my_accelerator', type=str, default='gpu')
    parser.add_argument('--my_max_epochs', type=int, default=200)
    parser.add_argument('--upload', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)    
    np.random.seed(args.seed)
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
                                         paths=paths,
                                         frac_train=args.train_fraction,
                                         select_subset=args.use_subset,
                                         random=args.random)

    # ------------
    # model
    # ------------
    n_feats = datamodule.get_n_feats()
    
    if args.wandb_pretrained is None:
        encoder = FCEncoder(16, args.embedding_size, args.d_model, n_feats, d_hidden = args.d_hidden, num_hidden_layers=args.num_hidden_layers)
        pmodel = CompressiveSensingPretraining(encoder, use_bn=False)
        model = GenotypeRegression(pmodel, lr=args.lr, 
                                   beta1=args.beta1, 
                                   beta2=args.beta2,
                                   factor=args.factor,
                                   monitor=args.monitor,
                                   l1_coef = args.l1_coef,
                                   freeze= args.freeze)
    else:
        
        run = wandb.init(mode="online",
                 project='genotype_pretraining', 
                 entity='chrisxx', 
                 job_type="inference",
                 dir=".",
                 settings=wandb.Settings(start_method='fork'))
        model_at = run.use_artifact("%s:latest"%args.wandb_pretrained)
        model_dir = model_at.download(root='%s/%s/'%(args.artifact_dir, args.wandb_pretrained))
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
        pmodel = CompressiveSensingPretraining.load_from_checkpoint('%s/%s/%s'%(args.artifact_dir, args.wandb_pretrained, mkey.split('/')[-1]), encoder=encoder)
        model = GenotypeRegression(pmodel, lr=args.lr, 
                                   beta1=args.beta1, 
                                   beta2=args.beta2,
                                   factor=args.factor,
                                   monitor=args.monitor,
                                   l1_coef = args.l1_coef,
                                   freeze= args.freeze)
        print(vars(pmodel))
    
    # ------------
    # wandb 
    # ------------
    wandb_logger = WandbLogger(entity=args.wandb_entity, 
                               project=args.wandb_project, 
                               name=args.wandb_name,
                               config=args)
    run = wandb_logger.experiment
    # save file to artifact folder
    
    result_dir = args.checkpoint_dir+'/%s/'%wandb_logger.experiment.name 
    os.makedirs(result_dir, exist_ok=True)
    copyfile(sys.argv[0], result_dir+sys.argv[0].split('/')[-1])
        
    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                          save_top_k=args.checkpoint_save_top_k,
                                          monitor=args.monitor)
    es_callback = EarlyStopping(monitor=args.monitor, 
                                mode=args.early_stopping_mode, 
                                patience=args.early_stopping_patience)
    lr_monitor = LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback,
                                                       es_callback, lr_monitor],
                                            log_every_n_steps=args.my_log_every_n_steps,
                                            accelerator=args.my_accelerator,
                                            max_epochs=args.my_max_epochs)
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
    if args.upload:
        print("uploading model...")
        #store config and model
        checkpoint_callback.to_yaml(checkpoint_callback.dirpath+'/checkpoint_callback.yaml')
        with open(checkpoint_callback.dirpath+'/config.yaml', 'w') as f:
            yaml.dump(run.config.as_dict(), f, default_flow_style=False)
        
        trained_model_artifact = wandb.Artifact(run.name, type="model", description="trained selfattn model")
        trained_model_artifact.add_dir(checkpoint_callback.dirpath)
        run.log_artifact(trained_model_artifact)


if __name__ == '__main__':
    main()
