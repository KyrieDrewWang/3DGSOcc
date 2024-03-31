#!/usr/bin/env bash
# CONFIG=$1
# GPUS=$2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=$((RANDOM + 10000))
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# python tools/train.py configs/renderocc/renderocc-7frame.py --gpu-id 8
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=8,4,5,7 python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \--nnodes=4 --master_port=8077 
#     $CONFIG \
#     --launcher pytorch ${@:3}
# CUDA_VISIBLE_DEVICES=8,4,5,6 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --master_port=8077 tools/train.py configs/renderocc/renderocc-7frame.py --launcher pytorch
CUDA_VISIBLE_DEVICES=1,2,3,8 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/renderocc/splattingocc-Nframe.py --launcher pytorch --work_dir splattingocc_workdir
