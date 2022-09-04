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
module load anaconda/2020.11-py3.8

source activate gsc
export PROJECT_DIR=/project/sds-rise/zhanwen/gsc
export DATA_DIR_VG_RCNN=/project/sds-rise/zhanwen/datasets
export SEED=1234
export NUM_GPUS=2
export MODEL_NAME="v1b4_sggen_riv_1"
export TORCHELASTIC_MAX_RESTARTS=0

echo "TRAINING SGGen model ${MODEL_NAME}"
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
mkdir ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
torchrun --nproc_per_node=$NUM_GPUS \
  ${PROJECT_DIR}/tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GSC True  \
  MODEL.ROI_RELATION_HEAD.USE_GSC_FE False  \
  SOLVER.TYPE SGD \
  SOLVER.IMS_PER_BATCH 128\
  SOLVER.MAX_ITER 30000 \
  SOLVER.BASE_LR 1e-3 \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL True \
  MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
  MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
  DTYPE "float32" \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.STEPS "(10000, 16000)" \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ${PROJECT_DIR}/datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ${PROJECT_DIR}/checkpoints/${MODEL_NAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
echo "Finished training SGGen model ${MODEL_NAME}"
