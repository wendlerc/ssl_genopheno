from argparse import ArgumentParser
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import numpy as np

from train import pretraining, training, evaluate, sklearn_evaluate
from modules import GCN, SparseClassifier, GBT_GCN, customGAT, customGIN

parser = ArgumentParser()

# experiment params
parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--model', default='GCN', type=str)
parser.add_argument('--path', default='best_model.pl', type=str)

parser.add_argument('--freeze', default='False', type=str)
parser.add_argument('--hid_dim', default=100, type=int)
parser.add_argument('--emb_dim', default=500, type=int)
parser.add_argument('--pretrain_lr', default=0.0001, type=float)
parser.add_argument('--train_lr', default=0.001, type=float)

parser.add_argument('--runs', default=10, type=int)
parser.add_argument('--pretrain_epochs', default=500, type=int)
parser.add_argument('--train_epochs', default=200, type=int)
parser.add_argument('--num_train_per_class', default=20, type=int)

args = parser.parse_args()


freeze = False if args.freeze == 'False' else True
emb_dim = args.emb_dim
hidden_dim = args.hid_dim
pretrain_lr = args.pretrain_lr
train_lr = args.train_lr
PATH = args.path
runs = args.runs
DATASET = args.dataset
MODEL = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l1_coefs = [0, 0.0001, 0.001, 0.01, 0.1, 1] 

for num_train_per_class in [1, 3, 7, 10, 13, 17, 20]:
    train_accs = torch.zeros((3, runs))
    val_accs = torch.zeros((3, runs))
    test_accs = torch.zeros((3, runs))

    for i in range(runs):
        ############# Dataset definition
        dataset = Planetoid(root='/tmp/{}'.format(DATASET), name=DATASET, num_train_per_class=num_train_per_class, split='random')
        data = dataset.data.to(device) # will load the whole dataset in one batch

        s_train = data.y[data.train_mask]
        s_val = data.y[data.val_mask]
        s_test = data.y[data.test_mask]

        for j, (embedding, augmentations) in enumerate([(True, True), (True, False), (False, False)]):
            train_acc = []
            val_acc = []
            val_loss = []
            test_acc = []

            ############# Pre-Training 
            epochs = args.pretrain_epochs
            model = GCN(dataset.num_node_features, hidden_dim, emb_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)

            if(embedding):
                pretraining(model, PATH, epochs, data, optimizer, freeze, s_train, s_val, s_test, 
                            dataset, hidden_dim, emb_dim, train_lr, l1_coefs, augmentations)
            
            ############# Training 
            epochs = args.train_epochs

            gcn = GCN(dataset.num_node_features, hidden_dim, emb_dim).to(device)
            model = SparseClassifier(torch.load(PATH) if embedding else gcn, emb_dim, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            if(not freeze):
                for k, C in enumerate(l1_coefs):
                    training(model, epochs, data, C, criterion, optimizer, s_train, s_val)

                    model = torch.load('best_net.pl')

                    evaluate(model, data, criterion, s_train, s_val, s_test, train_acc, val_acc, test_acc, val_loss)

                m = torch.argmax(torch.tensor(val_loss))
                print('Run {}, Embedding {}, Augmentations {}, val_acc {:.4f}, test_acc {:.4f}'.format(i, embedding, augmentations, val_acc[m], test_acc[m]))
                print('Best l1 reg coef is: {:.4f}'.format(l1_coefs[m]))

            else:
                for k, C in enumerate(l1_coefs):
                    if(embedding):
                        model = torch.load(PATH)
                    else:
                        model = GCN(dataset.num_node_features, hidden_dim, emb_dim).to(device)

                    sklearn_evaluate(model, data, C, s_train, s_val, s_test, train_acc, val_acc, test_acc)
                
                m = torch.argmax(torch.tensor(val_acc))
                print('Run {}, Embedding {}, Augmentations {}, val_acc {:.4f}, test_acc {:.4f}'.format(i, embedding, augmentations, val_acc[m], test_acc[m]))
                print('Best l1 reg coef is: {:.4f}'.format(l1_coefs[m]))
                
            train_accs[j, i] = train_acc[m]
            val_accs[j, i] = val_acc[m]
            test_accs[j, i] = test_acc[m]

    for j, (embedding, augmentations) in enumerate([('True', 'True'), ('True', 'False'), ('False', 'False')]):
        print('Avg train acc = {:.3f}+-{:.3f}'.format(torch.mean(train_accs[j]), 1.959960 * torch.std(train_accs[j]) / np.sqrt(runs)))
        print('Avg val acc = {:.3f}+-{:.3f}'.format(torch.mean(val_accs[j]), 1.959960 * torch.std(val_accs[j]) / np.sqrt(runs)))
        print('Avg test acc = {:.3f}+-{:.3f}'.format(torch.mean(test_accs[j]), 1.959960 * torch.std(test_accs[j]) / np.sqrt(runs)))
        print('Best test acc = {:.3f}'.format(torch.max(test_accs[j])))
        print('Worst test acc = {:.3f}'.format(torch.min(test_accs[j])))

        with open('results_{}_{}.csv'.format(MODEL, DATASET), 'a') as f:
            f.write('Frozen {}, Embedding {}, Augmentations {}'.format(args.freeze, embedding, augmentations))
            f.write('Num per class, avg train acc, val acc, test acc, best test acc, worst test acc \n')
            f.write('{}, {:.3f}+-{:.3f}, {:.3f}+-{:.3f}, {:.3f}+-{:.3f}, {:.3f},  {:.3f} \n'.format(num_train_per_class, torch.mean(train_accs[j]), 1.959960 * torch.std(train_accs[j]) / np.sqrt(runs),
                                                                                                torch.mean(val_accs[j]), 1.959960 * torch.std(val_accs[j]) / np.sqrt(runs),
                                                                                                torch.mean(test_accs[j]), 1.959960 * torch.std(test_accs[j]) / np.sqrt(runs),
                                                                                                torch.max(test_accs[j]), torch.min(test_accs[j])))