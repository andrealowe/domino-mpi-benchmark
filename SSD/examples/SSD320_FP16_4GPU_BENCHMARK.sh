# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CKPT_DIR=${1:-"/results/SSD320_FP16_4GPU"}
PIPELINE_CONFIG_PATH=${2:-"/workdir/models/research/configs"}"/ssd320_bench.config"
GPUS=4

TENSOR_OPS=0
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

mpirun --allow-run-as-root \
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
               --model_dir=${CKPT_DIR} \
               --alsologtostder \
               --amp \
               "${@:3}" 2>&1
               
PERF=$(echo "$TRAIN_LOG" | sed -n 's|.*global_step/sec: \(\S\+\).*|\1|p' | python -c "import sys; x = sys.stdin.readlines(); x = [float(a) for a in x[int(len(x)*3/4):]]; print(32*$GPUS*sum(x)/len(x), 'img/s')")

mkdir -p $CKPT_DIR
echo "$GPUS GPUs mixed precision training performance: $PERF" | tee $CKPT_DIR/train_log
echo "$TRAIN_LOG" >> $CKPT_DIR/train_log
