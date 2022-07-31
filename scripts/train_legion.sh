#!/usr/bin/env bash
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0 #3,4 #,4 #3,4
    export NUM_GPUS=1
    export MODEL_NAME="gbnet_vgg_1" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    echo "Started training PredCls model ${MODEL_NAME}"
    MODEL_DIRNAME=./checkpoints/${MODEL_NAME}/
    mkdir ${MODEL_DIRNAME} &&
    cp -r ./tools/ ${MODEL_DIRNAME} &&
    cp -r ./scripts/ ${MODEL_DIRNAME} &&
    cp -r ./maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
    HOST_NODE_ADDR=12345 PYTHONUNBUFFERED=x torchrun --master_port=17293 --nproc_per_node=$NUM_GPUS tools/relation_train_net.py \
    --config-file "configs/e2e_relation_detector_VGG16_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.PREDICTOR GBNetPredictor \
    MODEL.ROI_RELATION_HEAD.USE_GSC False  \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.TYPE 'FusedAdam' \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    TEST.IMS_PER_BATCH ${NUM_GPUS} \
    SOLVER.PRE_VAL False \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    DTYPE "float16" \
    SOLVER.MAX_ITER 32000 SOLVER.BASE_LR 1e-4 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT 'https://download.pytorch.org/models/vgg16-397923af.pth' \
    OUTPUT_DIR ${MODEL_DIRNAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
    # MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    # OUTPUT_DIR ./checkpoints/${MODEL_NAME}
    echo "Finished training PredCls model ${MODEL_NAME}"
    # MODEL.PRETRAINED_MODEL_CKPT /home/zhanwen/bpl_og/checkpoints/${MODEL_NAME}/model_0014000.pth \
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0 #3,4 #,4 #3,4
    export NUM_GPUS=1
    echo "TRAINING SGcls"
    MODEL_NAME="transformer_sgcls_dist15_2k_confmat_woInit"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10020 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GPUS \
    SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=0 # 5,6
    export NUM_GPUS=1 # 2
    echo "TRAINING SGdet"
    MODEL_NAME="transformer_sgdet_dist15_2k_confmat_woInit"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10021 --nproc_per_node=$NUM_GPUS \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferGSCPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 20 TEST.IMS_PER_BATCH $NUM_GPUS \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
fi
