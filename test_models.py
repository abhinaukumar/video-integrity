import numpy as np
from scipy.io import loadmat, savemat

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

import argparse
import time

def five_fold_cv(clf_name, x_data, y_data, n_trials = 500):
    
    '''
    Perform five-fold cross-validation.
    clf_name (string): Name of classifier - lda, lin_svm, gauss_svm, adaboost
    x_data (np.ndarray): Features to train classifier on. shape = (num_samples,num_features)
    y_data (np.ndarray): Target labels. shape = (num_samples,)
    n_trials (int): Number of times to repeat cross-validation (Default: 500)
    '''

    clf_dict = {'lda': LinearDiscriminantAnalysis, 'lin_svm': svm.SVC, 'gauss_svm': svm.SVC, 'adaboost': AdaBoostClassifier} # Classifiers considered
    clf_args = {'lda': {'solver':'svd'}, 'lin_svm': {'kernel': 'linear'}, 'gauss_svm': {'kernel':'rbf','gamma':'scale'}, 'adaboost': {'n_estimators':25}} # Arguments to classifiers

    
    med_train_acc = []
    med_test_acc = []

    for trial in range(n_trials):

        train_acc = []
        test_acc = []

        kf = KFold(n_splits = 5, shuffle = True)
        
        for train_inds, test_inds in kf.split(x_data):

            x_train = x_data[train_inds,:]
            y_train = y_data[train_inds]
            x_test = x_data[test_inds,:]
            y_test = y_data[test_inds]

            clf = clf_dict[clf_name](**clf_args[clf_name]) # Pass appropriate arguments to chosen classifier

            clf.fit(x_train,y_train)
            train_acc.append(np.sum(clf.predict(x_train) == y_train)/len(y_train))
            test_acc.append(np.sum(clf.predict(x_test) == y_test)/len(y_test))

        med_train_acc.append(np.median(train_acc))
        med_test_acc.append(np.median(test_acc))

    med_med_train_acc = np.median(med_train_acc)
    med_med_test_acc = np.median(med_test_acc)

    return [med_med_train_acc, med_med_test_acc]

parser = argparse.ArgumentParser(description='Code to test models on extracted features')

parser.add_argument('--model', help = 'Pretrained model used to extract embeddings (Default: alexnet)', default = 'alexnet')

args = parser.parse_args()

models = ['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'densenet121', 'squeezenet', 'mobilenetv1', 'mobilenetv2', 'all'] # Pretrained models considered

clf_list = ['lda', 'lin_svm', 'gauss_svm', 'adaboost'] # Classifiers considered

assert args.model in models, 'Invalid model name'

start = time.time()

if args.model != 'all':
    f = loadmat('data/features/l2/social_media_video_data_' + args.model +'.mat')
    x_data = f['x_data']
    y_data = f['y_data'].squeeze()

    unnorm_perf = {}
    norm_perf = {}
    all_perf = {}

    for clf_name in clf_list:
        unnorm_perf[clf_name] = five_fold_cv(clf_name, x_data[:,:6], y_data)
        norm_perf[clf_name] = five_fold_cv(clf_name, x_data[:,6:], y_data)
        all_perf[clf_name] = five_fold_cv(clf_name, x_data, y_data)

            
    print('Unnormalized Features:')
    print('Clasifier, Train acc, Test acc')

    for key in clf_list:
        print(key, ': ', unnorm_perf[key][0], unnorm_perf[key][1])

    print('Normalized Features:')
    print('Clasifier, Train acc, Test acc')

    for key in clf_list:
        print(key, ': ', norm_perf[key][0], norm_perf[key][1])

    print('All Features:')
    print('Clasifier, Train acc, Test acc')

    for key in clf_list:
        print(key, ': ', all_perf[key][0], all_perf[key][1])

else:
    
    unnorm_results = []
    norm_results = []
    all_results = []
    
    for clf_name in clf_list:
        unnorm_results.append([])
        norm_results.append([])
        all_results.append([])

    for model in models[:-1]:
        
        print("Testing pretrained model: "+model)
        f = loadmat('data/features/l2/social_media_video_data_' + model +'.mat')
        x_data = f['x_data']
        y_data = f['y_data'].squeeze()
        
        for clf_ind,clf_name in enumerate(clf_list):
            unnorm_results[clf_ind].append(five_fold_cv(clf_name, x_data[:,:6], y_data))
            norm_results[clf_ind].append(five_fold_cv(clf_name, x_data[:,6:], y_data))
            all_results[clf_ind].append(five_fold_cv(clf_name, x_data, y_data))

    results = np.array([unnorm_results, norm_results, all_results])

    savemat('results/l2/results.mat', {'results':results})

end = time.time()
print("Time elapsed: ", (end-start), "s")