bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 1 --d_model 512
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 2 --d_model 512
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 3 --d_model 512
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 4 --d_model 512
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 5 --d_model 512

bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 1 --d_model 512 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 2 --d_model 512 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 3 --d_model 512 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 4 --d_model 512 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv
bsub -o /cluster/scratch/wendlerc/lsfcsss/ -G es_puesch -n 12 -R "rusage[mem=1000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=8000]" -W 24:00 python project/genotype/pretrain.py --n_augs 5 --d_model 512 --path_pattern datasets/genotype/cas9/cas9_pairs_%s.csv




