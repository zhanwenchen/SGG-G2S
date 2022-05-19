#!/usr/bin/env bash
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS=2
    echo "Testing Predcls"
    MODEL_NAME="union_gsc_v1a_17.60_BPL_SA"
    MODEL_CHECKPOINT_NAME="model_0010000.pth"
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "checkpoints/${MODEL_NAME}/config.yml" \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        TEST.IMS_PER_BATCH $NUM_GPUS \
        DTYPE "float32" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  	    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
  	    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False;
    echo "Finished testing BPL SGDet model ${MODEL_NAME} at iteration ${MODEL_CHECKPOINT_NAME}";
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export NUM_GPUS=2
    MODEL_NAME="union_gsc_v1a_sgcls_1_model_0012000.pth_BPL"
    MODEL_CHECKPOINT_NAME="model_0012000.pth"
    echo "Testing SGCls model ${MODEL_NAME} at iteration ${MODEL_CHECKPOINT_NAME}"
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "checkpoints/${MODEL_NAME}/config.yml" \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        TEST.IMS_PER_BATCH $NUM_GPUS \
        DTYPE "float32" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False;
    echo "Finished testing BPL SGCls model ${MODEL_NAME} at iteration ${MODEL_CHECKPOINT_NAME}";
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=3,5
    export NUM_GPUS=2
    MODEL_NAME="union_gsc_v1a_sgdet_1_model_0016000.pth_BPL"
    MODEL_CHECKPOINT_NAME="model_0008000.pth"
    echo "Testing SGDet model ${MODEL_NAME} at iteration ${MODEL_CHECKPOINT_NAME}"
    python  -u  -m torch.distributed.launch --master_port 10055 --nproc_per_node=$NUM_GPUS \
        tools/relation_test_net.py \
        --config-file "checkpoints/${MODEL_NAME}/config.yml" \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        TEST.IMS_PER_BATCH $NUM_GPUS \
        DTYPE "float32" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
        MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME} \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
        TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False;
    echo "Finished testing BPL SGDet model ${MODEL_NAME} at iteration ${MODEL_CHECKPOINT_NAME}";
fi
