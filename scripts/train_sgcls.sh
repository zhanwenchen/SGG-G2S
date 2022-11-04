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
export TORCHELASTIC_MAX_RESTARTS=0
echo "TRAINING SGCls model ${MODEL_NAME}"
cd ${PROJECT_DIR}
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
fi
mkdir ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/.git/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
torchrun --nproc_per_node=$NUM_GPUS \
  ${PROJECT_DIR}/tools/relation_train_net.py \
  --config-file ${CONFIG_FILE} \
  MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_DATA ${PAIRWISE_METHOD_DATA} \
  MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_FUNC ${PAIRWISE_METHOD_FUNC} \
  MODEL.ROI_RELATION_HEAD.USE_GSC ${USE_GSC}  \
  MODEL.ROI_RELATION_HEAD.USE_GSC_FE ${USE_GSC_FE}  \
  SOLVER.TYPE SGD \
  SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.BASE_LR ${LR} \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
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
echo "Finished training SGCls model ${MODEL_NAME}" || echo "Failed to train SGCls model ${MODEL_NAME}"
