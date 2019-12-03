import argparse

import mxnet as mx
from mxnet import nd
from mxnet.gluon.model_zoo import vision
from gluoncv.data.transforms.presets.imagenet import transform_eval

from sklearn import metrics
import numpy as np
from scipy.io import savemat, loadmat
import cv2

import time
import matplotlib.pyplot as plt

plt.ion()

def extract_features(model, path):

    '''
    Extract embeddings from video.
    model (string): Name of pretrained model to use
    path (string): Path to video file
    '''

    model_dict = {'alexnet': vision.alexnet, \
            'vgg16': vision.vgg16_bn, \
            'vgg19': vision.vgg19_bn, \
            'resnet18': vision.resnet18_v2, \
            'resnet50': vision.resnet50_v2, \
            'densenet121': vision.densenet121, \
            'squeezenet': vision.squeezenet1_1, \
            'mobilenetv1': vision.mobilenet1_0, \
            'mobilenetv2': vision.mobilenet_v2_1_0} # Model classes

    ctx = mx.cpu()
    load_net = model_dict[model]
    net = load_net(pretrained=True, ctx=ctx)

    v = cv2.VideoCapture(path)

    feats = []
    k = 0
    while(v.isOpened() and k < 250): 
        ret,img = v.read() 
        if ret:
            if model == 'squeezenet':
                # Input size to squeezenet is different
                img = transform_eval(nd.array(img),resize_short=350,crop_size=299)
                # Squeezenet outputs 512 feature maps. Applying global pooling
                feats.append(np.mean(np.mean(net.features(img).asnumpy(),axis=-1),axis=-1))
            else:
                img = transform_eval(nd.array(img))
                feats.append(net.features(img).asnumpy())
            k += 1
        else: 
            break
    return feats

def extract_distances(feats):

    '''
    Extract spread and drift features.
    feats (np.ndarray): Array of embeddings. shape = (num_samples, num_dim)
    '''

    mean_feat = np.mean(feats,0)
    norm_mean_feat = np.linalg.norm(mean_feat)
    pair_dists = metrics.pairwise_distances(feats)
    mean_dists = np.linalg.norm(feats - mean_feat,axis=-1)
    adj_dists = [pair_dists[i,i-1] for i in range(1,pair_dists.shape[0])]

    # Four spread features
    sp1 = np.mean(mean_dists)
    sp2 = np.max(mean_dists)
    sp3 = np.mean(pair_dists)
    sp4 = np.max(pair_dists)

    # Two drift features
    dr1 = np.max(adj_dists)
    dr2 = np.mean(adj_dists)
    
    # Normalized spread features
    sp5 = np.mean(mean_dists)/norm_mean_feat
    sp6 = np.max(mean_dists)/norm_mean_feat
    sp7 = np.mean(pair_dists)/norm_mean_feat
    sp8 = np.max(pair_dists)/norm_mean_feat
    
    # Normalized drift features
    dr3 = dr1/norm_mean_feat
    dr4 = dr2/norm_mean_feat

    dist = np.expand_dims(np.array([sp1, sp2, sp3, sp4, dr1, dr2, sp5, sp6, sp7, sp8, dr3, dr4]), 0)

    return dist


parser = argparse.ArgumentParser(description='Code to extract features using pretrained models')

parser.add_argument('--path', help = 'Path to directory containing data', required = True)
parser.add_argument('--model', help = 'Pretrained model used to extract embeddings (Default: alexnet)', default = 'alexnet')

parser.add_argument('--num_static_videos', help = 'Number of static videos', default = 60)
parser.add_argument('--num_non_static_videos', help = 'Number of non-static videos', default = 60)

parser.add_argument('--save_embeddings', help = 'Flag to save framewise embeddings to a .mat file', action = 'store_true')
parser.add_argument('--load_embeddings', help = 'Flag to load framewise embeddings from a .mat file', action = 'store_true')
parser.add_argument('--save_features', help = 'Flag to save features to a .mat file', action = 'store_true')

args = parser.parse_args()

models = ['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'densenet121', 'squeezenet', 'mobilenetv1', 'mobilenetv2', 'all'] # Pretrained models considered

assert args.model in models, 'Invalid model name'
assert not(args.save_embeddings and args.load_embeddings), 'Cannot both load and save embeddings'

start = time.time()

if args.save_embeddings:

    static_feats = [] # Embeddings of static videos
    non_static_feats = [] # Embeddings of non-static videos
    static_frames = [] # Number of frames in each static video
    non_static_frames = [] # Number of frames in each non-static video

elif args.load_embeddings:
    f = loadmat('data/embeddings/video_embeddings_'+args.model+'.mat')

    # Load embeddings if already available

    static_feats = f['static_feats']
    non_static_feats = f['non_static_feats']
    static_frames = f['static_frames'].squeeze()
    non_static_frames = f['non_static_frames'].squeeze()


print("Processing Static Videos")

static_dists = []

for i in range(1,args.num_static_videos + 1):

    if not args.load_embeddings:
        feats = extract_features(args.model, args.path + 'static/' + str(i) +'.mp4')

        assert len(feats) > 50, 'Videos must have at least 51 frames'

        feats = np.vstack([feat.squeeze() for feat in feats])
        feats = feats[50:,:]
    else:
        feats = static_feats[static_frames[i-1]:static_frames[i]]
    
    if args.save_embeddings:
        static_feats.append(feats)
        static_frames.append(feats.shape[0])

    dist = extract_distances(feats)
    
    static_dists.append(dist)

print("Processing Non-static Videos")

non_static_dists = []

for i in range(1,args.num_non_static_videos + 1):

    if not args.load_embeddings:

        # Extract embeddings from pretrained model
        feats = extract_features(args.model, args.path + 'non_static/' + str(i) +'.mp4')

        assert len(feats) > 50, 'Videos must have at least 51 frames'

        # Collect all embeddings into an array
        feats = np.vstack([feat.squeeze() for feat in feats])
        feats = feats[50:,:]
    else:
        feats = non_static_feats[non_static_frames[i-1]:non_static_frames[i]]
    
    if args.save_embeddings:
        non_static_feats.append(feats)
        non_static_frames.append(feats.shape[0])

    dist = extract_distances(feats)
    
    non_static_dists.append(dist)

if args.save_embeddings:

    static_feats = np.concatenate(static_feats,axis=0)
    non_static_feats = np.concatenate(non_static_feats,axis=0)

    non_static_frames = np.cumsum(np.array(non_static_frames.insert(0,0)))
    static_frames = np.cumsum(np.array(static_frames.insert(0,0)))


x_data = np.concatenate(static_dists + non_static_dists,0)
y_data = np.concatenate([np.zeros((args.num_static_videos,)),np.ones((args.num_non_static_videos,))])

if args.save_features:
    savemat('data/features/l2/social_media_video_data_'+args.model+'.mat',{'x_data':x_data,'y_data':y_data})

if args.save_embeddings:
    savemat('data/embeddings/video_embeddings_'+args.model+'.mat',{'static_feats':static_feats,'non_static_feats':non_static_feats,'static_frames':static_frames,'non_static_frames':non_static_frames})

end = time.time()
print("Time elapsed: ", (end-start), "s")