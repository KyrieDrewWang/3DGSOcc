#!/bin/bash
##SBATCH --qos gpugpu
##SBATCH -N 1
##SBATCH -p vip_gpu_ailab 
##SBATCH -A ai4bio
##SBATCH --gres=gpu:4
##SBATCH --job-name 3-34

module load compilers/cuda/11.6 
module load cudnn/8.4.0.27_cuda11.x 
module load compilers/gcc/9.3.0 
module load llvm/triton-clang_llvm-11.0.1
source activate socc-v2

CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --master_port=29511 --nproc_per_node=4 tools/train.py configs/splattingocc/splattingocc-Nframe.py --work_dir &{exp_name} --launcher pytorch