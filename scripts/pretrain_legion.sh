#!/usr/bin/env bash
  export CUDA_VISIBLE_DEVICES=0 #3,4 #,4 #3,4
  export NUM_GPUS=1
  export MODEL_NAME="pretrain_vgg_backbone_fcs_1" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
  echo "Started pretraining model ${MODEL_NAME}"
  MODEL_DIRNAME=./checkpoints/pretrained_faster_rcnn/${MODEL_NAME}/
  mkdir ${MODEL_DIRNAME} &&
  cp -r ./tools/ ${MODEL_DIRNAME} &&
  cp -r ./scripts/ ${MODEL_DIRNAME} &&
  cp -r ./maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
  HOST_NODE_ADDR=12345 PYTHONUNBUFFERED=x torchrun --master_port=10001 --nproc_per_node=$NUM_GPUS  tools/detector_pretrain_net.py \
  --config-file "configs/pretrain_detector_VGG16_1x.yaml" \
  MODEL.VGG.PRETRAIN_STRATEGY fcs \
  MODEL.RELATION_ON False \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.TYPE 'FusedSGD' \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL True \
  DTYPE "float32" \
  SOLVER.MAX_ITER 100000 SOLVER.BASE_LR 1e-4 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.STEPS "(35000, 70000)" \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ./datasets/vg/ \
  OUTPUT_DIR ${MODEL_DIRNAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
  echo "Finished pretraining model ${MODEL_NAME}"
