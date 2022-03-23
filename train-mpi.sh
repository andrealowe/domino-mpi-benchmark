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

#edited for Domino

CKPT_DIR=${1:-"/mnt/SSD-results/multi-gpu"}
PIPELINE_CONFIG_PATH=${2:-"/mnt/SSD/configs/ssd320_bench.config"}
LOG_FILE=${3:-"train_log"}

mkdir -p $CKPT_DIR

pushd /mnt/SSD/models/research

python -u ./object_detection/model_main.py \
       --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
       --model_dir=${CKPT_DIR} \
       --alsologtostder \
       --amp \
       "${@:3}" 2>&1 | tee $CKPT_DIR/$LOG_FILE
