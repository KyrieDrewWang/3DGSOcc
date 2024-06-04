#!/bin/bash
##SBATCH --qos gpugpu
##SBATCH -N 2
##SBATCH --gres=gpu:4

module load compilers/cuda/11.6 
module load cudnn/8.4.0.27_cuda11.x 
module load compilers/gcc/9.3.0 
module load llvm/triton-clang_llvm-11.0.1
source activate socc-v2

#export LIBRARY_PATH=/HOME/xxxx/.conda/envs/py39/lib:$LIBRARY_PATH
#export LD_LIBRARY_PATH=/HOME/xxxx/.conda/envs/py39/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

### 获取每个节点的 hostname
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo ${host[$k]}
done

#python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等

#python3 setup.py build_ext --inplace


### 主节点运行
torchrun \
        --nnodes=2 \
        --node_rank=0 \
        --nproc_per_node=4 \
        --master_addr="${host[1]}" \
        --master_port="29501" \
 tools/train.py configs/renderocc/splattingocc-Nframe.py --launcher pytorch --work_dir socc_workdir \
 > train_rank0_${SLURM_JOB_ID}.log 2>&1 &
 
### 使用 srun 运行第二个节点
srun -N 1 --gres=gpu:4 -w ${host[2]} \
        torchrun \
        --nnodes=2 \
        --node_rank=1 \
        --nproc_per_node=4 \
        --master_addr="${host[1]}" \
        --master_port="29501" \
        tools/train.py configs/renderocc/splattingocc-Nframe.py --launcher pytorch --work_dir socc_workdir \
 >> train_rank1_${SLURM_JOB_ID}.log 2>&1  &
wait
