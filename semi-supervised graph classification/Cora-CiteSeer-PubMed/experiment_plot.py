from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('--plot', type=str, default='scatter', help='plot, scatter, barplot')
args = parser.parse_args()

filenames = ['results_Cora.csv', 'results_CiteSeer.csv', 'results_PubMed.csv']

if(args.plot == 'barplot'):
    for filename in filenames:
        with open(filename, 'r') as f:
            l1_coefs = []
            embed_true_freeze_false = []
            embed_false_freeze_false = []
            embed_true_freeze_true = []
            embed_false_freeze_true = []

            for line in f:
                info = line.split(',')
                if(info[0] == 'Frozen False ' and info[1] == ' Embedding True\n'):
                    config = embed_true_freeze_false
                elif(info[0] == 'Frozen False ' and info[1] == ' Embedding False\n'):
                    config = embed_false_freeze_false
                elif(info[0] == 'Frozen True ' and info[1] == ' Embedding True\n'):
                    config = embed_true_freeze_true
                elif(info[0] == 'Frozen True ' and info[1] == ' Embedding False\n'):
                    config = embed_false_freeze_true
                elif(info[0] == 'L1 Coefficient'):
                    continue
                else:
                    if(info[0] not in l1_coefs):
                        l1_coefs.append(info[0])

                    print(info)
                    test_acc = info[3].split('+-')
                    config.append([float(test_acc[0]), float(test_acc[1])])

        plt.figure()
        bins = l1_coefs
        ind = np.arange(len(bins)) 
        width = 0.2
        labels = ["Pretrained Embedding", "Embedding", "Frozen Pretrained Embedding", "Frozen Embedding" ]
        for k, data in enumerate([embed_true_freeze_false, embed_false_freeze_false, embed_true_freeze_true, embed_false_freeze_true]):
            data = np.array(data).T
            plt.bar(ind + k * width, data[0], width, label=labels[k], yerr=data[1])

        plt.xlabel("L1 regularization coef.")
        plt.ylabel("Average Test Accuracy")
        plt.xticks(ind + width, l1_coefs, rotation=70)
        plt.legend(loc='lower right')
        plt.savefig('barplot_{}.png'.format(filename))


elif(args.plot == 'train_fraction'):
    for filename in filenames:
        with open(filename, 'r') as f:
            train_fractions = []
            embed_true_freeze_false = []
            embed_false_freeze_false = []
            embed_true_freeze_true = []
            embed_false_freeze_true = []

            for line in f:
                info = line.split(',')
                if(info[0] == 'Frozen False ' and info[1] == ' Embedding True\n'):
                    config = embed_true_freeze_false
                elif(info[0] == 'Frozen False ' and info[1] == ' Embedding False\n'):
                    config = embed_false_freeze_false
                elif(info[0] == 'Frozen True ' and info[1] == ' Embedding True\n'):
                    config = embed_true_freeze_true
                elif(info[0] == 'Frozen True ' and info[1] == ' Embedding False\n'):
                    config = embed_false_freeze_true
                elif(info[0] == 'Num per class'):
                    continue
                else:
                    if(info[0] not in train_fractions):
                        train_fractions.append(info[0])

                    test_acc = info[3].split('+-')
                    config.append([float(test_acc[0]), float(test_acc[1])])

        plt.figure()
        labels = ["Pretrained Embedding", "Embedding", "Frozen Pretrained Embedding", "Frozen Embedding" ]
        for k, data in enumerate([embed_true_freeze_false, embed_false_freeze_false, embed_true_freeze_true, embed_false_freeze_true]):
            data = np.array(data).T
            plt.plot(train_fractions, data[0], label=labels[k])
            plt.fill_between(train_fractions, data[0] - data[1], data[0] + data[1], alpha=.1)

        plt.xlabel("Train samples per class")
        plt.ylabel("Average Test Accuracy")
        plt.xticks(train_fractions)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.savefig('train_fraction_plot_{}.png'.format(filename))