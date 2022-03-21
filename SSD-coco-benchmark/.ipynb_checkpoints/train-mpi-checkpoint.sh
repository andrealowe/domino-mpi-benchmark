#!/bin/bash

CKPT_DIR=${1:-"/mnt/SSD/results/SSD320_FP16_4GPU"}
PIPELINE_CONFIG_PATH=${2:-"/mnt/SSD/configs"}"/ssd320_full_4gpus.config"
MODEL_DIR="/domino/datasets/local/${DOMINO_PROJECT_NAME}/ssd/resnet_v1_50"
GPUS=4
 
TENSOR_OPS=0
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

pushd /mnt/SSD/models/research
 
time mpirun --allow-run-as-root \
       -np $GPUS \
       -H localhost:$GPUS \
       -bind-to none \
       -map-by slot \
       -x NCCL_DEBUG=INFO \
       -x LD_LIBRARY_PATH \
       -x PATH \
       -mca pml ob1 \
       -mca btl ^openib \
        python -u ./object_detection/model_main.py \
               --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
               --model_dir=${MODEL_DIR} \
               --checkpoint_dir=${CKPT_DIR} \
               --alsologtostder \
               --amp \
               "${@:3}"
               