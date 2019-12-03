import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import argparse

parser = argparse.ArgumentParser(description='Code to visualize performance of various classifiers')
parser.add_argument('--features', help = 'Features used to train the model - unnormalized, normalized or all (Default: unnormalized)', default = 'unnormalized')

args = parser.parse_args()

feat_dict = {'unnormalized': 0, 'normalized': 1, 'all': 2}

assert args.features in feat_dict.keys()

f = loadmat('results/l2/results.mat')
unnorm_results = f['results'][feat_dict[args.features]]

models = ['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'densenet121', 'squeezenet', 'mobilenetv1', 'mobilenetv2']

font = {'size'   : 12}

matplotlib.rc('font', **font)

for i in range(unnorm_results.shape[0]):

    train_acc = unnorm_results[i,:,0]
    test_acc = unnorm_results[i,:,1]

    x = np.arange(len(models))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, train_acc, width, label='Train accuracy')
    rects2 = ax.bar(x + width/2, test_acc, width, label='Test accuracy')

    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(top=1.4)
    ax.legend(loc=2, prop={'size': 22})

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.99)

    def autolabel(rects):
        ''' Attach a text label above each bar in *rects*, displaying its height.'''
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,4)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

fig.tight_layout()

plt.show()