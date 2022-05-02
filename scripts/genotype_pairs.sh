bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 1
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 2
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 3
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 4
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 5

bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 1 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 2 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 3 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 4 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 5 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv




