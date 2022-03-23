#!/bin/bash

GPUS=${1:-4}
file_loc=${2:-/mnt/SSD-results/multi-gpu/train_log}

sed -n 's|.*global_step/sec: \(\S\+\).*|\1|p' $file_loc | python -c "import sys; x = sys.stdin.readlines();  x = [float(a) for a in x[int(len(x)*3/4):]]; print(32*$GPUS*sum(x)/len(x), 'img/s')"

echo "$GPUS GPUs mixed precision training performance: $PERF" >> $file_loc 
