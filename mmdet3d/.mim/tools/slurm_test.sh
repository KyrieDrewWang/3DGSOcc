#!/usr/bin/env bash

# set -x

# PARTITION=$1
# JOB_NAME=$2
# CONFIG=$3
# CHECKPOINT=$4
# GPUS=${GPUS:-8}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# CPUS_PER_TASK=${CPUS_PER_TASK:-5}
# PY_ARGS=${@:5}
# SRUN_ARGS=${SRUN_ARGS:-""}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=8091 tools/test.py configs/renderocc/renderocc-7frame.py /data/wc/RenderOcc/splattingocc_workdir/epoch_12.pth --launcher pytorch --eval segm --gpu-collect #--tmpdir /home/wc/results/tmp  --dump_dir /home/wc/results 