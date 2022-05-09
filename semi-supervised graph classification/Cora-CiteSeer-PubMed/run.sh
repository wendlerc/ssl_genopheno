python lambda_plot.py --dataset Cora --freeze True  --runs 25
python lambda_plot.py --dataset Cora --freeze False --runs 25

python lambda_plot.py --dataset CiteSeer --freeze True --runs 25
python lambda_plot.py --dataset CiteSeer --freeze False --runs 25

python lambda_plot.py --dataset PubMed --freeze True --runs 25
python lambda_plot.py --dataset PubMed --freeze False --runs 25

python experiment_plot.py --plot train_fraction