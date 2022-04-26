#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:31:26 2022

@author: chrisw
"""


datasets = ["datasets/cfo/suite/bzip2d/bzip2d_1_LLVM_10_1000.csv",
            "datasets/cfo/suite/bzip2d/no_reps_bzip2d_1_LLVM_61_1000.csv",
            "datasets/cfo/suite/bzip2d/pow_no_reps_bzip2d_1_LLVM_61_1000.csv"]

encoders = ["fc", "selfattn", "transformer"]

bonus_flags = ["", "--no_augmentations"]


for d in datasets:
    for e in encoders:
        for b in bonus_flags:
            print("bsub -G es_puesch -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=8000]\"", end="")
            print(" -W 24:00 python project/cfo/augmented_pretrain.py --my_max_epochs 10000 --early_stopping_patience 50", end="")
            print(" --num_workers 12 --csv %s"%(d), end="")
            print(" --wandb_project cfo_pretrain_cluster --encoder %s %s"%(e, b))
            print("")
