#!/usr/bin/env bash
# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=$((RANDOM + 10000))
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# echo $MASTER_ADDR
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/test.py \
#     $CONFIG \
#     $CHECKPOINT \
#     --launcher pytorch \
#     --dump_dir results \
#     ${@:4}
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=8078 tools/test.py configs/renderocc/renderocc-7frame.py /data/wc/RenderOcc/work_dirs/renderocc-7frame/latest.pth --launcher pytorch --dump_dir results