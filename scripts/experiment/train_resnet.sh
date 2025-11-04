#!/bin/bash

#SBATCH --job-name=phi4-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source scripts/slurm/common.sh
source scripts/slurm/hf.sh
source scripts/slurm/deepspeed.sh

python -m src.main \
    cmd="train" \
    experiment=train_resnet