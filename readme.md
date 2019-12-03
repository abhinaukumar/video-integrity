## Video Integrity Testing using Minimal Learning

# Goal
Detect videos which are near-static, i.e., frames do not change (effectively) over time. Frames have added artefacts so the problem is non-trivial.

# Approach
Use deep networks trained on ImageNet to extract embeddings for each frame, and analyze the content of the video in this embedding space.

# Files
generate\_data.py - Extracts embeddings and features from a given directory of static and non-static videos. Assumes that the directory has two subdirectories - static and non-static, which each have videos numbered 1.mp4, 2.mp4 and so on.

test\_models.py - Tests various classifiers on the extracted features.

visualize\_results.py - Visualizes the performance of various classifiers on features extracted from various pretrained models as bar plots.

visualize\_embeddings.py - Visualizes the embeddings returned by a pretrained model as a connected 2D scatter plot.

run\_experiments.sh - Wrapper code to extract features from various pretrained models and test classifiers. Calls generate\_data.py and test\_models.py

# Dependencies
numpy
scipy
matplotlib

sklearn
mxnet
gluoncv

opencv
