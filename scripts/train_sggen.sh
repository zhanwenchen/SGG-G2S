export TORCHELASTIC_MAX_RESTARTS=0
export DATA_DIR_VG_RCNN=/project/sds-rise/zhanwen/datasets
export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "TRAINING SGGen model ${MODEL_NAME}"
cd ${PROJECT_DIR}
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
  SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.BASE_LR ${LR} \
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
  OUTPUT_DIR ${PROJECT_DIR}/checkpoints/${MODEL_NAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
echo "Finished training SGGen model ${MODEL_NAME}" || echo "Failed to train SGGen model ${MODEL_NAME}"
