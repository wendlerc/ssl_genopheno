bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_42 --seed 42
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_43 --seed 43
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_44 --seed 44
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_45 --seed 45
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_46 --seed 46
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_47 --seed 47
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_48 --seed 48
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_49 --seed 49
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_50 --seed 50
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 0 --wandb_name without_augmentations_51 --seed 51

bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_42 --seed 42
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_43 --seed 43
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_44 --seed 44
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_45 --seed 45
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_46 --seed 46
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_47 --seed 47
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_48 --seed 48
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_49 --seed 49
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_50 --seed 50
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --lr 0.001 --n_augs 1 --wandb_name with_augmentations_51 --seed 51






