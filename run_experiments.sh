#!/bin/bash

# List of pretrained models
models="alexnet vgg16 vgg19 resnet18 resnet50 densenet121 squeezenet mobilenetv1 mobilenetv2"

for model in $models
do

    echo "Creating data using ${model} and L2 distance"
    python3 data_gen.py --model $model --path /home/abhinau/Documents/databases/social_media_videos/ --load_embeddings --save_features

done;

echo "Testing classifiers on L2 features from ${model}"
python3 test_models.py --model all