# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:48:38 2022

@author: Chris Wendler
"""
import numpy as np

barlow_name = "ridgeslowverybig_with_augmentations"

pretrained = ["ridgeslowverybig_with_augmentations",
              "ridgeslowverybig_with_3augmentations",
              "ridgeslowverybig_with_6augmentations",
              "ridgeslowverybig_without_augmentations"]

for seed in (42 + np.arange(10)):
    print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G ls_krausea", end="")
    print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
    print(" -W 8:00 python project/genotype/regression.py --l1_coef 0 ", end="") 
    print(" --wandb_name trained_%d --seed %d --group trained"%(seed, seed))
    
    print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G ls_krausea", end="")
    print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
    print(" -W 8:00 python project/genotype/regression.py --l1_coef 0 --freeze", end="") 
    print(" --wandb_name random_%d --seed %d --group random"%(seed, seed))
    
    for pname in pretrained:
        print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G ls_krausea", end="")
        print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
        print(" -W 8:00 python project/genotype/regression.py --l1_coef 1 ", end="") 
        print(" --wandb_pretrained %s_%d"%(pname, seed), end="")
        print(" --wandb_name finetuned_%s_%d --seed %d --group finetuned_%s"%(pname, seed, seed, pname))
        
        print("bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G ls_krausea", end="")
        print(" -n 12 -R \"rusage[mem=1000, ngpus_excl_p=1]\" -R \"select[gpu_mtotal0>=10000]\"", end="")
        print(" -W 8:00 python project/genotype/regression.py --l1_coef 0.0085 --freeze", end="") 
        print(" --wandb_pretrained %s_%d"%(pname, seed), end="")
        print(" --wandb_name frozen_%s_%d --seed %d --group frozen_%s"%(pname, seed, seed, pname))