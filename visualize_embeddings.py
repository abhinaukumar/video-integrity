import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scipy.io import loadmat

import argparse
plt.ion()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

parser = argparse.ArgumentParser(description='Code to visualize one static and one non-static video\'s features from a dataset')
parser.add_argument('--model', help = 'Pretrained model used to extract embeddings (Default: alexnet)', default = 'alexnet')
parser.add_argument('--static_ids', help = 'IDs of the static video to be visualized', type = int, nargs = '*')
parser.add_argument('--non_static_ids', help = 'ID of the static video to be visualized', type = int, nargs = '*')
args = parser.parse_args()

assert len(args.static_ids) + len(args.non_static_ids) > 0, 'At least one video must be visualized'
assert len(args.static_ids) + len(args.non_static_ids) <= len(colors), 'At most ' + str(len(markers)) + ' videos can be visualized'

models = ['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'densenet121', 'squeezenet', 'mobilenetv1', 'mobilenetv2']

assert args.model in models, 'Invalid model name'

f = loadmat('data/embeddings/video_embeddings_'+args.model+'.mat')

static_feats = f['static_feats']
non_static_feats = f['non_static_feats']
static_frames = f['static_frames'].squeeze()
non_static_frames = f['non_static_frames'].squeeze()

all_feats = np.vstack([static_feats, non_static_feats])

pca = PCA(n_components=2)
pca_feats = pca.fit_transform(all_feats)

plt.figure()

for ind,vid_ind in enumerate(args.static_ids):

    static_video = pca_feats[static_frames[vid_ind - 1] : static_frames[vid_ind], :]

    # Plot 2D embeddings and connect with arrows
    plt.scatter(static_video[:,0], static_video[:,1], color = colors[ind], marker = 'o', label = 'Static Video ' + str(vid_ind))
    for i in range(1, static_video.shape[0]):
        plt.arrow(static_video[i-1,0],static_video[i-1,1], \
                static_video[i,0] - static_video[i-1,0], static_video[i,1] - static_video[i-1,1], \
                shape='full',color=colors[ind], lw=1.5, head_length=0.1, length_includes_head=True)

for ind,vid_ind in enumerate(args.non_static_ids):

    non_static_video = pca_feats[static_frames[-1] + non_static_frames[vid_ind - 1] : static_frames[-1] + non_static_frames[vid_ind], :]

    # Plot 2D embeddings and connect with arrows
    plt.scatter(non_static_video[:,0], non_static_video[:,1], color = colors[len(args.static_ids) + ind], marker = '*', label = 'Non-static Video ' + str(vid_ind))
    for i in range(1, non_static_video.shape[0]):
        plt.arrow(non_static_video[i-1,0],non_static_video[i-1,1], \
                non_static_video[i,0] - non_static_video[i-1,0], non_static_video[i,1] - non_static_video[i-1,1], \
                shape='full', color=colors[len(args.static_ids) + ind], lw=1.5, head_length=0.1, length_includes_head=True)

plt.legend(loc=2, prop={'size': 20})