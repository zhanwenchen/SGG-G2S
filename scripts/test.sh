#!/usr/bin/env bash
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=2,3
    export NUM_GPUS=2
    export MODEL_NAME="v1b4_gscfe_2"
    MODEL_CHECKPOINT_NAME="model_0012000.pth"
    echo "Testing Predcls model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}"
    python  -u  -m torch.distributed.launch --master_port 20035 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "checkpoints/${MODEL_NAME}/config.yml" \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        TEST.IMS_PER_BATCH $NUM_GPUS \
        DTYPE "float32" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False;
    echo "Testing Predcls model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}";
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=3,5
    export NUM_GPUS=2
    MODEL_NAME="union_gsc_v1a_sgcls_1"
    MODEL_CHECKPOINT_NAME="model_0004000.pth"
    echo "Testing SGCls model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}"
    python  -u  -m torch.distributed.launch --master_port 10041 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        TEST.IMS_PER_BATCH $NUM_GPUS DTYPE "float32" \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        TEST.ALLOW_LOAD_FROM_CACHE True TEST.VAL_FLAG False;
    echo "Finished testing SGCls model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}";
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS=2
    MODEL_NAME="union_gsc_v1a_sgdet_1"
    MODEL_CHECKPOINT_NAME="model_0004000.pth"
    echo "Testing SGDet model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}"
    python  -u  -m torch.distributed.launch --master_port 10040 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        TEST.IMS_PER_BATCH $NUM_GPUS DTYPE "float32" \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False;
    echo "Finished testing SGDet model ${MODEL_NAME} at ${MODEL_CHECKPOINT_NAME}";
fi
