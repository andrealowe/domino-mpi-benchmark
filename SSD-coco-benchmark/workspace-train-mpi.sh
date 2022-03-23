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

mpirun \
    -wdir /mnt/SSD-coco-benchmark/SSD/models/research \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 \
    -mca btl ^openib \
    python -u ./object_detection/model_main.py \
           --pipeline_config_path="/mnt/SSD-coco-benchmark/SSD/configs/ssd320_bench.config" \
           --model_dir="/mnt/SSD-coco-benchmark/results/multi-gpu" \
           --amp \
           | tee /mnt/SSD-coco-benchmark/results/multi-gpu/train_log
       