<div align="center">    
 
# Self-supervised Learning meets Compressive Sensing

</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@gitlab.inf.ethz.ch:PRV-PUESCHEL/wendlerc/csss.git

# install project   
cd csss
conda create -n csss python=3.8
conda activate csss
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#pip install -e .   
pip install -r requirements.txt
conda install jupyter
```   
 Next, run experiment using.   
 ```bash
python project/cfo/regression.py
```
or
 ```bash
python project/cfo/pretrain.py
```

Useful command for starting crashed jobs on the cluster:

cat scripts/genotype_evaluation_nol1.sh | grep frozen.*without.*51 | sh
