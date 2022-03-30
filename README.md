<div align="center">    
 
# Self-supervised Learning meets Compressive Sensing

</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@github.com:chrislybaer/selfsupervisedcs.git

# install project   
cd selfsupervisedcs
conda create -n csss python=3.8
conda activate csss
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_mnist.py --gpus 1
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from project.lit_mnist import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=128)
val_loader = DataLoader(mnist_val, batch_size=128)
test_loader = DataLoader(mnist_test, batch_size=128)

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
