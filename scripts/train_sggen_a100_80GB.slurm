#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH -J pytorchtest
#SBATCH -o pytorchtest-%A.out
#SBATCH -e pytorchtest-%A.err
#SBACTH --exclude=udc-an28-1,udc-an28-7

module purge
module load singularity

export SEED=1234
export MODEL_NAME="v1b4_sggen_riv_1"
export BATCH_SIZE=128
export MAX_ITER=30000
export LR=1e-3
export PROJECT_DIR=/project/sds-rise/zhanwen/gsc

singularity exec --nv docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/train_sggen.sh
