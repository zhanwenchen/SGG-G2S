#!/usr/bin/env bash
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GPUS=1
    MODEL_NAME_OLD="union_only_v2b_1" # TODO: CHANGE
    STARTING_WEIGHTS="model_final.pth" # TODO: CHANGE
    MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_continue"
    echo "TRAINING Predcls model ${MODEL_NAME_NEW}"
    [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
    mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    \
    MODEL.WEIGHT ./checkpoints/${MODEL_NAME_OLD}/${STARTING_WEIGHTS} \
    SOLVER.BASE_LR 5e-4 \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.MAX_ITER 32000 \
    SOLVER.STEPS "(10000, 16000, 20000, 24000, 26000, 27000, 32000)" \
    SOLVER.PRE_VAL True \
    \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    TEST.IMS_PER_BATCH ${NUM_GPUS} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    DTYPE "float32" \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME_NEW};
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=1,2 #3,4 #,4 #3,4
    export NUM_GPUS=2
    MODEL_NAME_OLD="v1a_predcls_17.60" # TODO: CHANGE
    STARTING_WEIGHTS="model_0014000.pth" # TODO: CHANGE
    MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_continue_sgcls"
    echo "TRAINING SGCls model ${MODEL_NAME_NEW}"
    [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
    mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    python -u -m torch.distributed.launch --master_port 10030 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.MAX_ITER 24000 \
    SOLVER.BASE_LR 1e-3 \
    SOLVER.STEPS "(10000, 16000, 20000, 24000)" \
    SOLVER.PRE_VAL True \
    \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    TEST.IMS_PER_BATCH $NUM_GPUS \
    DTYPE "float32" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
  export CUDA_VISIBLE_DEVICES=1,2 #3,4 #,4 #3,4
  export NUM_GPUS=2
  MODEL_NAME_OLD="v1a_predcls_17.60" # TODO: CHANGE
  STARTING_WEIGHTS="model_0014000.pth" # TODO: CHANGE
  MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_continue_sgdet"
  echo "TRAINING SGDet model ${MODEL_NAME_NEW}"
  [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
  mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
  cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
  cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
  cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
  python -u -m torch.distributed.launch --master_port 10030 --nproc_per_node=$NUM_GPUS \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
  \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.MAX_ITER 24000 \
  SOLVER.BASE_LR 1e-3 \
  SOLVER.STEPS "(10000, 16000, 20000, 24000)" \
  SOLVER.PRE_VAL True \
  \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  TEST.IMS_PER_BATCH $NUM_GPUS \
  DTYPE "float32" \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
  MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
fi
