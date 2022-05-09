#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:48:38 2022

@author: Chris Wendler
"""
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--project', default="genotype_supervised_smallerdata_l1eq0", type=str)
parser.add_argument('--l1_coef', default=0, type=float)
parser.add_argument('--l1_coef_fixed', default=0, type=float)
parser.add_argument('--share', default="ls_krausea", type=str)

args = parser.parse_args()


project = args.project
pretrained = ["verybig_6mixed"]




for seed in (42 + np.arange(10)):    
    for frac in [0.05, 0.10, 0.25, 0.5, 1.]:
        for pname in pretrained:
            print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G %s"%args.share, end="")
            print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
            print(" -W 4:00 python project/genotype/regression.py --l1_coef %2.4f "%args.l1_coef, end="") 
            print(" --wandb_pretrained %s_%d"%(pname, seed), end="")
            print(" --wandb_project %s"%project, end="")
            print(" --use_subset --train_fraction %2.4f"%frac, end="")
            print(" --wandb_name finetuned_%s_%d --seed %d --group finetuned_%s"%(pname, seed, seed, pname))
            
            print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G %s"%args.share, end="")
            print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
            print(" -W 4:00 python project/genotype/regression.py --l1_coef %2.4f --freeze"%args.l1_coef_fixed, end="") 
            print(" --wandb_pretrained %s_%d"%(pname, seed), end="")
            print(" --wandb_project %s"%project, end="")
            print(" --use_subset --train_fraction %2.4f"%frac, end="")
            print(" --wandb_name frozen_%s_%d --seed %d --group frozen_%s"%(pname, seed, seed, pname))