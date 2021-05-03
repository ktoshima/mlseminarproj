#!/bin/bash

#SBATCH -J H_v_t
#SBATCH -p gpu
#SBATCH -N 1                    # number of nodes
#SBATCH -n 16                    # number of cores
#SBATCH --gres=gpu:1
#SBATCH --mem 128000              # memory pool for all cores
#SBATCH -t 0-08:00              # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o Job.%N.%j.out # STDOUT
#SBATCH -e Job.%N.%j.err # STDERR

module load Anaconda3/5.0.1-fasrc01 cuda/10.1.243-fasrc01 cudnn/7.6.5.32_cuda10.1-fasrc01
source activate tf-gpu
python train_unet_only.py > logs/log_train-unet-only_$(date "+%Y.%m.%d-%H.%M.%S").txt 2>&1
