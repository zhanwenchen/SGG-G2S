#/bin/bash

# For non-Slurm servers.

# Assumption: you have already cloned this repository

export ENV_NAME=gsc
export INSTALL_DIR=${HOME}
export DATASET_URL=https://sgg-zhanwen.s3.amazonaws.com/datasets.zip
export DATASETS_DIR=${HOME}/datasets
export PROJECT_DIR=${HOME}/gsc


# TODO: assert that checkpoints and log directories don't exist.


echo "Step 1: Installing dependencies (binaries)"
conda create --name ${ENV_NAME} python=3.8 ipython scipy h5py pandas -y
conda activate ${ENV_NAME}

# quantization depends on pytorch=1.10 and above
# torchvision: https://github.com/pytorch/vision/releases/tag/v0.11.3
# torchaudio: https://github.com/pytorch/audio/releases/tag/v0.10.2
# 1.11.0 changes C++ API so DeviceUtils will be removed and csrc will need an update
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y


# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python-headless overrides tensorboard comet_ml torchinfo setuptools==59.5.0


# install pycocotools
cd ${INSTALL_DIR}
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
# NOTE: you must have access to the target GPU for CUDA architecture detection.
# Set up git: use ssh-keygen and add it to the
cd ${INSTALL_DIR}
git clone https://github.com/zhanwenchen/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext


# install project code
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
# if you have't downloaded it, do
# git clone https://github.com/zhanwenchen/SGG-G2S.git gsc
cd ${PROJECT_DIR} && git checkout relation_augmentation && python setup.py build develop

unset INSTALL_DIR


echo "Step 2: Downloading Data"
parentdir="$(dirname "${DATASETS_DIR}")"
cd ${parentdir}
wget ${DATASET_URL}
unzip datasets.zip
cd ${DATASETS_DIR}/glove
rm glove.6B.200d.txt


echo "Step 3: Test Training"

cd ${PROJECT_DIR}
mkdir checkpoints
mkdir log
ln -s ${DATASETS_DIR}/pretrained_faster_rcnn ${PROJECT_DIR}/checkpoints

wget https://gist.githubusercontent.com/zhanwenchen/13aed95aea596de82382ea1671079beb/raw/8177993891a7e424ed053bda007aa16e149b505e/my_secrets.py -O maskrcnn_benchmark/utils/my_secrets.py
