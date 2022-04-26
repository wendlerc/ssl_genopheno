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

learning_rates = [0.001, 0.0001, 0.00001]

num_layers = [3, 6, 12]

i = 1
for d in datasets:
    for e in encoders:
        for b in bonus_flags:
            for lr in learning_rates:
                for nl in num_layers:
                    print("echo %d;"%i)
                    print("bsub -G ls_krausea -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=8000]\"", end="")
                    print(" -W 24:00 python project/cfo/augmented_pretrain.py --my_max_epochs 100000 --early_stopping_patience 100", end="")
                    print(" --num_workers 12 --csv %s --lr %2.8f --num_layers %d"%(d, lr, nl), end="")
                    print(" --wandb_project cfo_pretrain_cluster_sweep --encoder %s %s;"%(e, b))
                    i += 1
