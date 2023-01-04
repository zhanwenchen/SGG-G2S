#!/bin/bash

timestamp() {
  date +"%Y%m%d%H%M%S"
}

SLURM_JOB_NAME=44699200_vctree_pairwise_sgcls_4GPU_riv_1
SLURM_JOB_ID=0012000

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

export PROJECT_DIR=${HOME}/gsc
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
export WEIGHT=${PROJECT_DIR}/checkpoints/${SLURM_JOB_NAME}/model_${SLURM_JOB_ID}.pth
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/


if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  export CUDA_VISIBLE_DEVICES=0
  export SEED=1234
  export BATCH_SIZE=12
  export MAX_ITER=50000
  export LR=1e-3
  export USE_GSC=False
  export USE_GSC_FE=False
  export PAIRWISE_METHOD_DATA='hadamard'
  export PAIRWISE_METHOD_FUNC='mha'
  export USE_PAIRWISE_L2=True
  export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml
  export DATA_DIR_VG_RCNN=${HOME}/datasets
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c); ((NUM_GPUS++))
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=True
  export PRE_VAL=False
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
  export WITH_CLEAN_CLASSIFIER=True
  export WITH_TRANSFER_CLASSIFIER=True
  export NUM2AUG=4
  export MAX_BATCHSIZE_AUG=32

  ${PROJECT_DIR}/scripts/train_vctree.sh
fi
