#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4 # need to match number of gpus
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB # need to match batch size.
#SBATCH -J pytorchtest
#SBATCH -o /home/pct4et/gsc/log/pytorchtest-%A.out
#SBATCH -e /home/pct4et/gsc/log/pytorchtest-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu
#SBACTH --exclude=udc-an28-1,udc-an28-7

timestamp() {
  date +"%Y-%m-%d %T"
}

error_exit()
{
#   ----------------------------------------------------------------
#   Function for exit due to fatal program error
#       Accepts 1 argument:
#           string containing descriptive error message
#   Source: http://linuxcommand.org/lc3_wss0140.php
#   ----------------------------------------------------------------
    echo "$(timestamp) ERROR ${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    echo "$(timestamp) ERROR ${PROGNAME}: Exiting Early."
    exit 1
}

error_check()
{
#   ----------------------------------------------------------------
#   This function simply checks a passed return code and if it is
#   non-zero it returns 1 (error) to the calling script.  This was
#   really only created because I needed a method to store STDERR in
#   other scripts and also check $? but also leave open the ability
#   to add in other stuff, too.
#
#   Accepts 1 arguments:
#       return code from prior command, usually $?
#  ----------------------------------------------------------------
    TO_CHECK=${1:-0}

    if [ "$TO_CHECK" != '0' ]; then
        return 1
    fi

}

export PROJECT_DIR=/home/pct4et/gsc
export MODEL_NAME="v1b4_sggen_riv_1"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}_${SLURM_JOB_ID}.log
else
  module purge
  module load singularity

  export SEED=1234
  export BATCH_SIZE=256
  export MAX_ITER=30000
  export LR=1e-3
  export SINGULARITYENV_PREPEND_PATH="/home/pct4et/.conda/envs/gsc_docker/bin:/opt/conda/condabin"

  singularity exec --nv --env LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/home/pct4et/.conda/envs/gsc_docker/lib" docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/train_sggen.sh
fi
