#/bin/bash

# For non-Slurm servers.

# Assumption: you have already cloned this repository

export ENV_NAME=gsc_docker
export INSTALL_DIR=${HOME}
export CONDA_PREFIX=/scratch/pct4et/envs

# TODO: assert that checkpoints and log directories don't exist.

cd ${INSTALL_DIR}
git clone https://github.com/cocodataset/cocoapi.git
git clone git@github.com:zhanwenchen/apex.git

singularity shell --nv docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

export ENV_NAME=gsc_docker
export PROJECT_DIR=${HOME}/gsc
export INSTALL_DIR=${HOME}
export ENVS_DIR=/scratch/pct4et/envs
mkdir -p ${ENVS_DIR}

echo "Step 1: Installing dependencies (binaries)"
conda create -p ${ENVS_DIR}/${ENV_NAME} python=3.8 ipython scipy h5py pandas -y
source activate ${ENV_NAME}

# torchvision: https://github.com/pytorch/vision/releases/tag/v0.13.1 for PyTorch 1.12.1
# torchaudio: https://github.com/pytorch/audio/releases/tag/v0.12.1
# 1.11.0 changes C++ API so DeviceUtils will be removed and csrc will need an update
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y


# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python-headless overrides tensorboard setuptools==59.5.0


# install pycocotools
cd ${INSTALL_DIR}/cocoapi/PythonAPI
python setup.py build_ext install

# install apex
# NOTE: you must have access to the target GPU for CUDA architecture detection.
cd ${INSTALL_DIR}/apex

 # from singularity
exit && ijob -A sds-rise -p gpu --gres=gpu:a100:1 -c 16 --mem=32000

singularity shell --nv docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
export INSTALL_DIR=${HOME}
export ENV_NAME=gsc_docker
export PROJECT_DIR=${HOME}/gsc
source activate ${ENV_NAME}
cd ${INSTALL_DIR}
cd apex
python setup.py install --cuda_ext --cpp_ext

# install project code
cd ${PROJECT_DIR}

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
