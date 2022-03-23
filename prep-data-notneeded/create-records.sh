#!/bin/bash

COCO_DIR=/domino/datasets/local/${DOMINO_PROJECT_NAME}/coco
mkdir -p $COCO_DIR/coco2017_tfrecords
PYTHONPATH="/mnt/SSD/models/research/:/mnt/SSD/models/research/slim/:$PYTHONPATH"

TRAIN_ANNOTATIONS_FILE="${COCO_DIR}/annotations/instances_train2017.json"
VAL_ANNOTATIONS_FILE="${COCO_DIR}/annotations/instances_val2017.json"
TESTDEV_ANNOTATIONS_FILE="${COCO_DIR}/annotations/image_info_test-dev2017.json"
TRAIN_IMAGE_DIR="${COCO_DIR}/train2017"
TEST_IMAGE_DIR="${COCO_DIR}/test2017"
VAL_IMAGE_DIR="${COCO_DIR}/val2017"
OUTPUT_DIR="${COCO_DIR}/coco2017_tfrecords"

pushd /mnt/SSD/models/research

python object_detection/dataset_tools/create_coco_tf_record.py \
  --logtostderr \
  --include_masks \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"