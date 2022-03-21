# MPI Benchmark

Tests are adapted from the [NVIDIA product performance tests](https://developer.nvidia.com/deep-learning-performance-training-inference)

---

## Training with synthetic data

### Environments

#### MPI client/workspace environment

- Base Image URI : ```nvcr.io/nvidia/tensorflow:22.02-tf2-py3```

- Dockerfile Instructions
**Ensure 'Automatically make compatible with Domino' is checked.**

#### MPI cluster compute environment

- Base Image URI : ```nvcr.io/nvidia/tensorflow:22.02-tf2-py3```
- Cluster Type: MPI

### MPI workspace settings - DFS project

#### Environment and Hardware

- **Workspace Environment**: _The above MPI client/workspace environment_
- **Workspace IDE**: JupyterLab
- **Hardware Tier**: Small


#### Compute Hardware

- **Attach Compute Cluster**: MPI
- **Number of Workers**: Between 2-8
- **Worker Hardware Tier**: GPU (small)
- **Cluster Compute Environment**: _The above MPI cluster compute environment_

### Steps to set up the project, workspace, and test data

1. Create a project with the name "mpi-benchmark".
2. Start a workspace with the above configuration.
3. Execute the following code:

```
mpirun python -u /workspace/nvidia-examples/cnn/resnet.py --batch_size 256 
--num_iter 500 --precision fp16 --iter_unit batch
```

4. Use the throughput (images/sec) from the last epoch as the indicator of
  performance. A single AWS EC2 T4 GPU was shown to process 429 images/sec.
  80% efficiency is desirable when looking at the cluster size (the num of
  GPUs x the single GPU rate)

---

## Horovod Benchmarks

Test adapted from the [Horovod docs](https://horovod.readthedocs.io/en/stable/benchmarks_include.html)

### Environments

#### MPI client/workspace environment

- Base Image URI : ```horovod/horovod```

- Dockerfile Instructions
**Ensure 'Automatically make compatible with Domino' is checked.**

#### MPI cluster compute environment

- Base Image URI : ```horovod/horovod```
- Cluster Type: MPI

### MPI workspace settings - DFS project

#### Environment and Hardware

- **Workspace Environment**: _The above MPI client/workspace environment_
- **Workspace IDE**: JupyterLab
- **Hardware Tier**: Small


#### Compute Hardware

- **Attach Compute Cluster**: MPI
- **Number of Workers**: Between 2-8
- **Worker Hardware Tier**: GPU (small)
- **Cluster Compute Environment**: _The above MPI cluster compute environment_

### Steps to set up the project, workspace, and test data

1. Create a project with the name "horovod-benchmark".
2. Start a workspace with the above configuration.
3. Execute the following code:

```
git clone https://github.com/tensorflow/benchmarks
cd benchmarks

mpirun \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model resnet101 \
        --batch_size 64 \
        --variable_update horovod
```

4. At the end of the run you will see the number of images processed per second. 

---

## Training with COCO dataset

### Environments

#### MPI client/workspace environment

- Base Image URI : ```nvcr.io/nvidia/tensorflow:22.02-tf1-py3```

- Dockerfile Instructions
**Ensure 'Automatically make compatible with Domino' is checked.**
 Then add the following:
```
RUN git clone https://github.com/NVIDIA/DeepLearningExamples

WORKDIR DeepLearningExamples/TensorFlow/Detection/SSD

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libpmi2-0-dev \
 && rm -rf /var/lib/apt/lists/*

RUN PROTOC_VERSION=3.0.0 && \
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    rm -f $PROTOC_ZIP

RUN pip install -r requirements.txt
RUN pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'

WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH="/mnt/SSD/models/research/:/mnt/SSD/models/research/slim/:/mnt/SSD/models/research/object_detection/:$PYTHONPATH"
```

#### MPI cluster compute environment

- Base Image URI : ```nvcr.io/nvidia/tensorflow:22.02-tf1-py3```

- Cluster Type: MPI

- Dockerfile Instructions
```
RUN git clone https://github.com/NVIDIA/DeepLearningExamples

WORKDIR DeepLearningExamples/TensorFlow/Detection/SSD

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libpmi2-0-dev \
 && rm -rf /var/lib/apt/lists/*

RUN PROTOC_VERSION=3.0.0 && \
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    rm -f $PROTOC_ZIP

RUN pip install -r requirements.txt
RUN pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'

WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH="/mnt/SSD/models/research/:/mnt/SSD/models/research/slim/:/mnt/SSD/models/research/object_detection/:$PYTHONPATH"
```

### MPI workspace settings - DFS project

#### Environment and Hardware

- **Workspace Environment**: _The above MPI client/workspace environment_
- **Workspace IDE**: JupyterLab
- **Hardware Tier**: Medium


#### Compute Hardware

- **Attach Compute Cluster**: MPI
- **Number of Workers**: 4
- **Worker Hardware Tier**: GPU (small)
- **Cluster Compute Environment**: _The above MPI cluster compute environment_

### Steps to set up the project, workspace, and test data

1. Create a project with the name "mpi-ml-benchmark".
2. Start a workspace
3. Git clone or otherwise copy the benchmark repo to your workpace/project.
   * [mpi-benchmark/](https://github.com/andrealowe/domino-mpi-benchmark/tree/main/SSD-coco-benchmark)
4. Copy the test data to the project's default dataset directory
   ```
   aws s3 cp s3://mpi-test-coco-data /domino/datasets/local/mpi-ml-benchmark --region us-west-2 --recursive --no-sign-request
   ```

### Running the benchmark script

1. Start a workspace with the above configuration.
2. Execute the mpi-train.sh script:

```
/.mpi-train.sh
```

To compare to a single GPU, execute the ```train.sh``` script.

NVIDIA found an increase to 549 images/sec with 8xT4 GPUs from 98 images/sec for 1 T4 GPU. 
