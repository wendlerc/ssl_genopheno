# All features are frozen
# python graph_features.py --dataset Cora  --runs 25
# python graph_features.py --dataset CiteSeer  --runs 25
# python graph_features.py --dataset PubMed  --runs 25

# All features are un-frozen
# python semi-supervised.py --dataset Cora --freeze False --runs 25
# python semi-supervised.py --dataset CiteSeer --freeze False --runs 25
# python semi-supervised.py --dataset PubMed --freeze False --runs 25

# python GraphBarlow.py --dataset Computers
# python GraphBarlow.py --dataset CS
# python GraphBarlow.py --dataset Physics
# python GraphBarlow.py --dataset Photo

# python GraphBarlow.py --dataset Cora --model GAT  --runs 25
# python GraphBarlow.py --dataset CiteSeer  --model GAT --runs 25
# python GraphBarlow.py --dataset PubMed  --model GAT --runs 25

python GraphBarlow.py --dataset Cora --model GIN  --runs 25
python GraphBarlow.py --dataset CiteSeer  --model GIN --runs 25
python GraphBarlow.py --dataset PubMed  --model GIN --runs 25

# python semi-supervised.py --dataset Cora --model GAT --freeze False --runs 25
# python semi-supervised.py --dataset CiteSeer --model GAT --freeze False --runs 25
# python semi-supervised.py --dataset PubMed --model GAT --freeze False --runs 25
