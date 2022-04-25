python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=10 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name fc --encoder fc
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=10 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name transformer --encoder transformer
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=10 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name selfattn --encoder selfattn

python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=30 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name fc --encoder fc
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=30 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name transformer --encoder transformer
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=30 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name selfattn --encoder selfattn

python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=61 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name fc --encoder fc
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=61 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name transformer --encoder transformer
python project/cfo/regression.py --path_pattern 'datasets/cfo/tierry_splits/%s_n=61 approximate_dataset.csv' --wandb_project cfo_tierry_splits --wandb_name selfattn --encoder selfattn
