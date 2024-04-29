#!/bin/bash
module load compilers/cuda/11.6 
module load cudnn/8.4.0.27_cuda11.x 
module load compilers/gcc/9.3.0 
module load llvm/triton-clang_llvm-11.0.1
source activate socc-v2

export PYTHONUNBUFFERED=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# Using async gradient all reduce 
export CUDA_DEVICE_MAX_CONNECTIONS=1

### nodes gpus rank master_addr job_id
# nodes
NODES=$1
# gpus
NPROC_PER_NODE=$2

# rank
NODE_RANK=$3

# master
MASTER_ADDR=$4
MASTER_PORT="29301"

#JOB ID
BATCH_JOB_ID=$5

# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
OUTPUT_LOG="train_rank${NODE_RANK}_${BATCH_JOB_ID}.log"

WORK_DIR="test"

torchrun \
     --nnodes="${NODES}" \
     --node_rank="${NODE_RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
     tools/train.py \
     "configs/renderocc/splattingocc-Nframe.py" \
     --launcher pytorch \
     --work_dir $WORK_DIR >> "${OUTPUT_LOG}" 2>&1
