#!/usr/bin/env bash
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=3,5
    export NUM_GPUS=2
    MODEL_NAME_OLD="union_gsc_v1a_sgdet_1" # TODO: CHANGE
    STARTING_WEIGHTS="model_0016000.pth" # TODO: CHANGE
    MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_BPL"
    echo "TRAINING Predcls with BPL: ${MODEL_NAME_NEW}"
    [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
    mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GPUS \
        tools/relation_train_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
        \
        SOLVER.BASE_LR 5e-4 \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.MAX_ITER 24000 \
        SOLVER.STEPS "(10000, 16000, 20000, 24000)" \
        SOLVER.PRE_VAL True \
        \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        TEST.IMS_PER_BATCH ${NUM_GPUS} \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        DTYPE "float32" \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.PRETRAINED_MODEL_CKPT ./checkpoints/${MODEL_NAME_OLD}/${STARTING_WEIGHTS} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME_NEW};
    echo "Finished training BPL PredCls model ${MODEL_NAME_NEW}";
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS=2
    MODEL_NAME_OLD="union_gsc_v1a_sgcls_1" # TODO: CHANGE
    STARTING_WEIGHTS="model_0012000.pth" # TODO: CHANGE
    MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_BPL"
    echo "TRAINING SGCls with BPL: ${MODEL_NAME_NEW}"
    [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
    mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    python -u -m torch.distributed.launch --master_port 10034 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
        \
        SOLVER.BASE_LR 5e-4 \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.MAX_ITER 32000 \
        SOLVER.STEPS "(10000, 16000, 20000, 24000, 32000)" \
        SOLVER.PRE_VAL True \
        \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        TEST.IMS_PER_BATCH ${NUM_GPUS} \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        DTYPE "float32" \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.PRETRAINED_MODEL_CKPT ./checkpoints/${MODEL_NAME_OLD}/${STARTING_WEIGHTS} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME_NEW};
    echo "Finished training BPL PredCls model ${MODEL_NAME_NEW}";
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=3,5
    export NUM_GPUS=2
    MODEL_NAME_OLD="union_gsc_v1a_sgdet_1" # TODO: CHANGE
    STARTING_WEIGHTS="model_0016000.pth" # TODO: CHANGE
    MODEL_NAME_NEW="${MODEL_NAME_OLD}_${STARTING_WEIGHTS}_BPL"
    echo "TRAINING SGDet with BPL: ${MODEL_NAME_NEW}"
    [ -d "./checkpoints/${MODEL_NAME_OLD}" ] &&
    mkdir ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/tools/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./scripts/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    cp -r ./checkpoints/${MODEL_NAME_OLD}/maskrcnn_benchmark/ ./checkpoints/${MODEL_NAME_NEW}/ &&
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
        \
        SOLVER.BASE_LR 5e-4 \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.MAX_ITER 24000 \
        SOLVER.STEPS "(10000, 16000, 20000, 24000)" \
        SOLVER.PRE_VAL True \
        \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        TEST.IMS_PER_BATCH ${NUM_GPUS} \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        DTYPE "float32" \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.PRETRAINED_MODEL_CKPT ./checkpoints/${MODEL_NAME_OLD}/${STARTING_WEIGHTS} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME_NEW};
    echo "Finished training BPL PredCls model ${MODEL_NAME_NEW}";
fi
