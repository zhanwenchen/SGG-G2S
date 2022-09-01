#!/bin/bash
#SBATCH -A sds-rise
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -x udc-an28-[1,7]
#SBATCH -c 4
#SBATCH -t 24:00:00 # TODO
#SBATCH -J pytorchtest
#SBATCH -o pytorchtest-%A.out
#SBATCH -e pytorchtest-%A.err

module purge
module load anaconda/2020.11-py3.8

source activate gsc
cd /scratch/pct4et/gsc
scripts/train_legion.sh 2
