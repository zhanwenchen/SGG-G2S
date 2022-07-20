#!/usr/bin/env bash
  export CUDA_VISIBLE_DEVICES=0 #3,4 #,4 #3,4
  export NUM_GPUS=1
  export MODEL_NAME="pretrain_vgg_1" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
  echo "Started pretraining model ${MODEL_NAME}"
  MODEL_DIRNAME=./checkpoints/pretrained_faster_rcnn/${MODEL_NAME}/
  mkdir ${MODEL_DIRNAME} &&
  cp -r ./tools/ ${MODEL_DIRNAME} &&
  cp -r ./scripts/ ${MODEL_DIRNAME} &&
  cp -r ./maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
  HOST_NODE_ADDR=12345 PYTHONUNBUFFERED=x torchrun --master_port=10001 --nproc_per_node=$NUM_GPUS  tools/detector_pretrain_net.py \
  --config-file "configs/pretrain_detector_VGG16_1x.yaml" \
  MODEL.RELATION_ON False \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.TYPE 'FusedSGD' \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL False \
  DTYPE "float16" \
  SOLVER.MAX_ITER 50000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.STEPS "(30000, 45000)" \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT 'https://download.pytorch.org/models/vgg16-397923af.pth' \
  OUTPUT_DIR ${MODEL_DIRNAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
  # MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  # OUTPUT_DIR ./checkpoints/${MODEL_NAME}
  echo "Finished pretraining model ${MODEL_NAME}"
  # MODEL.PRETRAINED_MODEL_CKPT /home/zhanwen/bpl_og/checkpoints/${MODEL_NAME}/model_0014000.pth \
