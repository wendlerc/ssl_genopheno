import torch
from utils import augment
from loss import CompressiveSensingLoss, barlow_twins_loss
import sklearn.linear_model as lm
from modules import SparseClassifier, GCN
from xgboost import XGBClassifier
import torch.nn as nn
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def pretraining(model, PATH, epochs, data, optimizer, freeze, s_train, s_val, s_test, 
                dataset=None, hidden_dim=None, emb_dim=None, train_lr=None, l1_coefs=None, augmentations=False, px=0.1, pe=0.5):
    criterion = CompressiveSensingLoss()

    early = 100
    best = 0

    for epoch in range(epochs):
        if augmentations:
            data1 = augment(data, px=px, pe=pe)
            data2 = augment(data, px=px, pe=pe)
            features1, features2 = model(data1), model(data2)
            loss = barlow_twins_loss(features1, features2)
        else:
            features = model(data)
            loss = criterion(features)    
        loss.backward()
        optimizer.step()

        if(not freeze):
            if(epoch % 20 == 0):
                val = validate_pretrain(model, data, dataset, emb_dim, train_lr, l1_coefs, s_train, s_val, s_test)
                print('Best Evaluated accuracy on validation is {:.3f}'.format(val))

        else:
            if(epoch % 10 == 0):
                val = sklearn_validate_pretrain(model, data, dataset, emb_dim, train_lr, l1_coefs, s_train, s_val, s_test)
                print('Best Evaluated accuracy on validation is {:.3f}'.format(val))
            
        if val > best:
            best = val
            early = 100
            torch.save(model, PATH)
        else:
            early -= 1
            if early == 0:
                break

    # if(not freeze):
    #     torch.save(model, PATH)


def training(model, epochs, data, C, criterion, optimizer, s_train, s_val):
    early = 10
    best = 10

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output[data.train_mask], s_train)
        loss += C * torch.norm(model.linear_classifier.weight, p=1) # ||r_hat||_1 Regularization for sparsity
        loss.backward()
        optimizer.step()    

        val_loss = criterion(output[data.val_mask], s_val) 

        if val_loss < best:
            best = val_loss
            early = 20
            torch.save(model, 'best_net.pt')
        else:
            early -= 1
            if early == 0:
                break


def evaluate(model, data, criterion, s_train, s_val, s_test, train_acc=None, val_acc=None, test_acc=None, val_loss=None, validate=False):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        if (not validate):
            output = model(data) 
            val_loss.append( criterion(output[data.train_mask], s_train) )
            train_acc.append( (pred[data.train_mask] == s_train).sum() / int(data.train_mask.sum()) )
            val_acc.append( (pred[data.val_mask] == s_val).sum() / int(data.val_mask.sum()) )
            test_acc.append( (pred[data.test_mask] == s_test).sum() / int(data.test_mask.sum()) )
            # nonzero_vals[j, k, i] = torch.count_nonzero(torch.where(model.linear_classifier.weight > 0.001, 1, 0)).cpu()
        else:
            return (pred[data.val_mask] == s_val).sum() / int(data.val_mask.sum()) 


def evaluate_graph_features(features, data, C, s_train, s_val, s_test, train_acc=None, val_acc=None, test_acc=None):
    est = lm.LogisticRegression(penalty='l1', solver='liblinear', C=1/C) if C != 0 else lm.LogisticRegression(penalty='none', solver='lbfgs')
    est.fit(features[data.train_mask].cpu().detach(), s_train.cpu().ravel())
    train_acc.append( est.score(features[data.train_mask].cpu().detach(), s_train.cpu().ravel()) )
    val_acc.append( est.score(features[data.val_mask].cpu().detach(), s_val.cpu().ravel()) )
    test_acc.append( est.score(features[data.test_mask].cpu().detach(), s_test.cpu().ravel()) )
    print('Number of nonzero coefs is {}'.format(features.shape[1]))


def sklearn_evaluate(model, data, C, s_train, s_val, s_test, train_acc=None, val_acc=None, test_acc=None, validate=False):
    model.eval()
    with torch.no_grad():
        features = model(data)
        est = lm.LogisticRegression(penalty='l1', solver='liblinear', C=1/C) if C != 0 else lm.LogisticRegression(penalty='none', solver='lbfgs')
        est.fit(features[data.train_mask].cpu().detach(), s_train.cpu().ravel())

        if(not validate):
            train_acc.append( est.score(features[data.train_mask].cpu().detach(), s_train.cpu().ravel()) )
            val_acc.append( est.score(features[data.val_mask].cpu().detach(), s_val.cpu().ravel()) )
            test_acc.append( est.score(features[data.test_mask].cpu().detach(), s_test.cpu().ravel()) )
            print('Number of nonzero coefs is {} and l1 reg coef is {}'.format((est.coef_ != 0).sum(), C))
        else: 
            return est.score(features[data.val_mask].cpu().detach(), s_val.cpu().ravel())


def validate_pretrain(gcn, data, dataset, emb_dim, train_lr, l1_coefs, s_train, s_val, s_test):
    embedding = copy.deepcopy(gcn)
    model = SparseClassifier(embedding, emb_dim, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0
    for k, C in enumerate(l1_coefs):
        training(model, 200, data, C, criterion, optimizer, s_train, s_val)

        model = torch.load('best_net.pt')

        val = evaluate(model, data, criterion, s_train, s_val, s_test, validate=True)

        if(val > best):
            best = val
    
    return best

def sklearn_validate_pretrain(gcn, data, dataset, emb_dim, train_lr, l1_coefs, s_train, s_val, s_test):
    model = copy.deepcopy(gcn)

    best = 0
    for C in l1_coefs:
        val = sklearn_evaluate(model, data, C, s_train, s_val, s_test, validate=True)

        if(val > best):
            best = val
    
    return best

