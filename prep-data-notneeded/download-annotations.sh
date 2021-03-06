#!/bin/bash
 
# Get COCO 2017 data sets
pushd /domino/datasets/local/${DOMINO_PROJECT_NAME}
mkdir -p coco
pushd /domino/datasets/local/${DOMINO_PROJECT_NAME}/coco

curl -O https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
tar xzf coco_annotations_minival.tgz
 
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
 
curl -O http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip
 
rm coco_annotations_minival.tgz annotations_trainval2017.zip 
 
# TBD: MD5 verification
# $md5sum *.zip *.tgz
#f4bbac642086de4f52a3fdda2de5fa2c  annotations_trainval2017.zip
#cced6f7f71b7629ddf16f17bbcfab6b2  train2017.zip
#442b8da7639aecaf257c1dceb8ba8c80  val2017.zip
#2d2b9d2283adb5e3b8d25eec88e65064  coco_annotations_minival.tgz
 
popd