#!/bin/bash

file_loc=${1:-/mnt/SSD-coco-benchmark/results/single-gpu/train_log}
GPUS=${2:-1}

sed -n 's|.*global_step/sec: \(\S\+\).*|\1|p' $file_loc | python -c "import sys; x = sys.stdin.readlines();  x = [float(a) for a in x[int(len(x)*3/4):]]; print(32*$GPUS*sum(x)/len(x), 'img/s')"

echo "$GPUS GPU mixed precision training performance: $PERF" >> $file_loc
