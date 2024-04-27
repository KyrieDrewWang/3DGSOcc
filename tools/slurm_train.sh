#!/usr/bin/env bash

PARTITION=ai4bio
JOB_NAME=occ_training
CONFIG=configs/renderocc/splattingocc-Nframe.py
WORK_DIR=socc_base
GPUS=16
GPUS_PER_NODE=4
CPUS_PER_TASK=8

set -x

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --nodes=1 \
    --gres=gpu:${GPUS_PER_NODE} \
    # --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" 
