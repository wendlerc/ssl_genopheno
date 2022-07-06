from argparse import ArgumentParser
from torch_geometric.datasets import Planetoid
import torch
import numpy as np
import torch.nn as nn

from utils import dataset_subset, load_dataset, show_mat
from train import training, pretraining, evaluate, sklearn_evaluate, evaluate_graph_features
from modules import SparseClassifier, GBT_GCN, customGAT, customGIN

parser = ArgumentParser()

# experiment params
parser.add_argument('--dataset', default='WikiCS', type=str)
parser.add_argument('--model', default='GCN', type=str)
parser.add_argument('--path', default='best_model.pt', type=str)
parser.add_argument('--frozen', default='False', type=str) # choices True, all, false: all means also graph Features
parser.add_argument('--orth', default='True', type=str)

parser.add_argument('--hid_dim', default=100, type=int)
parser.add_argument('--emb_dim', default=512, type=int) # must be greater than train set for compressive sensing matrix setting
parser.add_argument('--pretrain_lr', default=0.0005, type=float)
parser.add_argument('--train_lr', default=0.001, type=float)

parser.add_argument('--runs', default=20, type=int)
parser.add_argument('--pretrain_epochs', default=500, type=int)
parser.add_argument('--num_train_per_class', default=20, type=int)

args = parser.parse_args()

emb_dim = args.emb_dim
orthogonality_inspection = True if args.orth == 'True' else False
hidden_dim = args.hid_dim
pretrain_lr = args.pretrain_lr
train_lr = args.train_lr
PATH = args.path
runs = args.runs
DATASET = args.dataset
MODEL = args.model
frozen = True if args.frozen == 'True' else False

settings =  [(True, True, False), (True, False, False), (False, False, False)]
print_settings = [('True', 'True', 'False'), ('True', 'False', 'False'), ('False', 'False', 'False')]

if frozen == 'True':
    settings += [(False, False, True), (False, False, True)]
    print_settings += [('False', 'False', 'True'), ('False', 'False', 'True')]

config_augmentations = {
    'Cora' : (0.1, 0.5),
    'CiteSeer': (0.1, 0.5),
    'PubMed': (0.1, 0.5),
    'WikiCS': (0.1, 0.5),
    'Computers': (0.1, 0.4),
    'Photo': (0.3, 0.1),
    'CS': (0.2, 0.5),
    'Physics': (0.2, 0.5)
}
px, pe = config_augmentations[DATASET]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################  Fixed Fourier features

if (DATASET in ['Cora', 'CiteSeer']): # Pubmed impossible to compute eigendecomposition
    dataset = Planetoid(root='/tmp/{}'.format(DATASET), name=DATASET)
    data = dataset.data.to(device) 

    e = data.edge_index
    N = data.x.shape[0]
    A = torch.zeros((N, N)).to(device)
    for i in range(e.shape[1]):
        A[e[0,i], e[1,i]] = 1
    A = torch.where(A + A.T > 0, 1., 0.)

    _, adjacency_features = torch.linalg.eig(A)
    adjacency_features = adjacency_features.real[:, :50]

    L = torch.diag(A.sum(dim=1)) - A
    _, laplacian_features = torch.linalg.eig(L)
    laplacian_features = laplacian_features.real[:, :50]

else: 
    adjacency_features, laplacian_features = None, None


l1_coefs = [0, 0.0001, 0.001, 0.01, 0.1, 1] 
train_fractions = [1, 3, 7, 10, 13, 17, 20] if DATASET in ['Cora', 'CiteSeer', 'PubMed'] else [1/10, 2/10, 4/10, 5/10, 7/10, 10/10]
L = len(settings)

for num_train_per_class in train_fractions:
    train_accs = torch.zeros((L, runs))
    val_accs = torch.zeros((L, runs))
    test_accs = torch.zeros((L, runs))

    if (DATASET in ["WikiCS", "Computers", "Photo", "CS", "Physics"]):
        dataset, data = load_dataset(DATASET)
        train_mask = dataset_subset(data, num_train_per_class, runs=runs).to(device)
        val_mask = data.val_mask
        test_mask = data.test_mask

    for i in range(runs):
        ############# Dataset definition
        if (DATASET in ["WikiCS", "Computers", "Photo", "CS","Physics"]):
            data.train_mask = train_mask[:, i]
            data.val_mask = val_mask[:, i]
            data.test_mask = test_mask if DATASET == 'WikiCS' else test_mask[:, i]

        else:
            dataset = Planetoid(root='/tmp/{}'.format(DATASET), name=DATASET, num_train_per_class=num_train_per_class, split='random')
            data = dataset.data.to(device) # will load the whole dataset in one batch
            train, val, test = data.train_mask, data.val_mask, data.test_mask
        
        s_train = data.y[data.train_mask]
        s_val = data.y[data.val_mask]
        s_test = data.y[data.test_mask]
        
        for j, (embedding, augmentations, graph_features) in enumerate(settings):
            train_acc = []
            val_acc = []
            val_loss = []
            test_acc = []

            ############# Pre-Training 
            epochs = args.pretrain_epochs

            # defining encoder
            if (MODEL == 'GCN'):
                encoder = GBT_GCN(dataset.num_node_features, emb_dim, batch_normalization = True).to(device)
            elif (MODEL == 'GIN'):
                encoder = customGIN(dataset.num_node_features, emb_dim).to(device)
            else :
                encoder = customGAT(dataset.num_node_features, emb_dim).to(device)

            if(embedding):
                # pretraining the encoder
                optimizer = torch.optim.Adam(encoder.parameters(), lr=pretrain_lr)
                pretraining(encoder, PATH, epochs, data, optimizer, True, s_train, s_val, s_test, 
                            dataset, hidden_dim, emb_dim, train_lr, l1_coefs, augmentations, px, pe)

            if(orthogonality_inspection and embedding and not augmentations):
                # visualizing how well the encoder satisfies the orthogonality criterion
                show_mat(encoder, data, MODEL, DATASET, embedding, emb_dim)


            ############# Training 
            if(not frozen):
                ### Case of training the whole architecture during finetuning -- requires PyTorch
                for k, C in enumerate(l1_coefs):
                    classifier =  SparseClassifier(torch.load(PATH) if embedding else encoder, emb_dim, dataset.num_classes).to(device)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=pretrain_lr)
                    criterion = nn.CrossEntropyLoss()

                    training(classifier, epochs, data, C, criterion, optimizer, s_train, s_val) # saves best model at best_net.pt

                    classifier = torch.load('best_net.pt')
                    evaluate(classifier, data, criterion, s_train, s_val, s_test, train_acc, val_acc, test_acc, val_loss)

                m = torch.argmax(torch.tensor(val_loss))
                print('Run {}, Embedding {}, Augmentations {}, val_acc {:.4f}, test_acc {:.4f}'.format(i, embedding, augmentations, val_acc[m], test_acc[m]))
                print('Best l1 reg coef is: {:.4f}'.format(l1_coefs[m]))

            else: 
                ### Case of using frozen features and training only the classifier during finetuning -- we choose to use sklearn
                for k, C in enumerate(l1_coefs):
                    ## Case where features are extracted from graph matrices
                    if(graph_features):
                        if(j == 3):
                            features = adjacency_features
                        else:
                            features = laplacian_features

                        if (features is None): # this is every case except Cora, Citeseer where eigendecomposition is possible because of small dataset size
                            train_acc, val_acc, test_acc = [0], [0], [0]
                        else:
                            evaluate_graph_features(features, data, C, s_train, s_val, s_test, train_acc, val_acc, test_acc)
                    else:
                        ## cases where features are extracted from a GNN encoder (pretrained or not)
                        if(embedding):
                            encoder = torch.load(PATH) 
                        else:
                            if (MODEL == 'GCN'):
                                encoder = GBT_GCN(dataset.num_node_features, emb_dim, batch_normalization = True).to(device)
                            elif (MODEL == 'GIN'):
                                encoder = customGIN(dataset.num_node_features, emb_dim).to(device)
                            else :
                                encoder = customGAT(dataset.num_node_features, emb_dim).to(device)
                        sklearn_evaluate(encoder, data, C, s_train, s_val, s_test, train_acc, val_acc, test_acc)
                
                    m = torch.argmax(torch.tensor(val_acc))
                    print('Run {}, Embedding {}, Augmentations {}, val_acc {:.4f}, test_acc {:.4f}, graph features {}, {}'.format(
                        i, embedding, augmentations, val_acc[m], test_acc[m], 'True' if graph_features else 'False', '' if j < 3 else 'adjacency' if j==3 else 'laplacian'))
                    print('Best l1 reg coef is: {:.4f}'.format(l1_coefs[m]))
                
            train_accs[j, i] = train_acc[m]
            val_accs[j, i] = val_acc[m]
            test_accs[j, i] = test_acc[m]

    for j, (embedding, augmentations, graph_features) in enumerate(print_settings):
        print('Avg train acc = {:.3f}+-{:.3f}'.format(torch.mean(train_accs[j]), 1.959960 * torch.std(train_accs[j]) / np.sqrt(runs)))
        print('Avg val acc = {:.3f}+-{:.3f}'.format(torch.mean(val_accs[j]), 1.959960 * torch.std(val_accs[j]) / np.sqrt(runs)))
        print('Avg test acc = {:.3f}+-{:.3f}'.format(torch.mean(test_accs[j]), 1.959960 * torch.std(test_accs[j]) / np.sqrt(runs)))
        print('Best test acc = {:.3f}'.format(torch.max(test_accs[j])))
        print('Worst test acc = {:.3f}'.format(torch.min(test_accs[j])))

        with open('results_{}_{}.csv'.format(MODEL, DATASET), 'a') as f:
            f.write(' Embedding {}, Freeze {}, Augmentations {}, graph features {}, {}'.format(embedding, frozen, augmentations, graph_features, '' if j < 3 else 'adjacency' if j==3 else 'laplacian'))
            f.write('Num per class, avg train acc, val acc, test acc, best test acc, worst test acc \n')
            f.write('{}, {:.3f}+-{:.3f}, {:.3f}+-{:.3f}, {:.3f}+-{:.3f}, {:.3f},  {:.3f} \n'.format(num_train_per_class, torch.mean(train_accs[j]), 1.959960 * torch.std(train_accs[j]) / np.sqrt(runs),
                                                                                                torch.mean(val_accs[j]), 1.959960 * torch.std(val_accs[j]) / np.sqrt(runs),
                                                                                                torch.mean(test_accs[j]), 1.959960 * torch.std(test_accs[j]) / np.sqrt(runs),
                                                                                                torch.max(test_accs[j]), torch.min(test_accs[j])))