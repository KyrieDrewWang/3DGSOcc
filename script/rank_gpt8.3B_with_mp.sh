#!/bin/bash
module load anaconda/2021.11
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load cudnn/8.4.0.27_cuda11.x
source activate Megatron-LM

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
MASTER_PORT="29501"

#JOB ID
BATCH_JOB_ID=$5

# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
OUTPUT_LOG="train_rank${NODE_RANK}_${BATCH_JOB_ID}.log"

CHECKPOINT_PATH="checkpoints/gpt2_8.3b_${BATCH_JOB_ID}"
VOCAB_FILE="gpt2-vocab.json"
MERGE_FILE="gpt2-merges.txt"
DATA_PATH="enwiki-gpt2_text_document"

torchrun \
     --nnodes="${NODES}" \
     --node_rank="${NODE_RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
     pretrain_gpt.py \
     --tensor-model-parallel-size 1 \
     --pipeline-model-parallel-size 4 \
     --sequence-parallel \
     --num-layers 72 \
     --hidden-size 3072 \
     --num-attention-heads 32 \
     --micro-batch-size 1 \
     --global-batch-size 16 \
     --seq-length 1024 \
     --max-position-embeddings 1024 \
     --train-iters 500000 \
     --lr-decay-iters 320000 \
     --save $CHECKPOINT_PATH \
     --load $CHECKPOINT_PATH \
     --data-path $DATA_PATH \
     --vocab-file $VOCAB_FILE \
     --merge-file $MERGE_FILE \
     --data-impl mmap \
     --split 949,50,1 \
     --distributed-backend nccl \
     --lr 0.00015 \
     --min-lr 1.0e-5 \
     --lr-decay-style cosine \
     --weight-decay 1e-2 \
     --clip-grad 1.0 \
     --lr-warmup-fraction .01 \
     --checkpoint-activations \
     --log-interval 100 \
     --save-interval 10000 \
     --eval-interval 1000 \
     --eval-iters 10 >> "${OUTPUT_LOG}" 2>&1
