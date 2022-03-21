#!/bin/bash
 
# Get Resnet checkpoints
pushd /domino/datasets/local/${DOMINO_PROJECT_NAME}
mkdir -p ssd
pushd /domino/datasets/local/${DOMINO_PROJECT_NAME}/ssd

wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzf resnet_v1_50_2016_08_28.tar.gz
rm resnet_v1_50_2016_08_28.tar.gz

mkdir -p resnet_v1_50
mv resnet_v1_50.ckpt resnet_v1_50/model.ckpt
