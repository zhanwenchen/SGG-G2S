# ijob -A sds-rise -p gpu --gres=gpu:a100:1 -c 12
conda create -n gsc python=3.8 # use gsc_a100 for example, but maybe even node-specific
conda activate gsc
# conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# CUDA 11.4 Not working for Pytorch 1.11.0. Try 1.10.1.
# The only way 1.10.1 works is if you install it first before anything else.
# Verify with import torch; torch.cuda.is_available(). Should be true.
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install ipython ninja yacs cython matplotlib tqdm h5py
# TODO: build h5py with parallel support

pip install opencv-python setuptools==59.5.0 scipy tensorboard comet-ml

mkdir ~/gsc_install && cd ~/gsc_install
export INSTALL_DIR=$(pwd)
echo $INSTALL_DIR

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
module load gcc/9.2.0
module load cuda/11.4.2
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# TODO: Edit out the raise RuntimeError in setup.py if there's a minor version mismatch.

# install PyTorch Detection
git clone git@github.com:zhanwenchen/SGG-G2S.git gsc
cd gsc
git checkout v1b4
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
# download csrc folder from cpp_update branch because it is only there.
python setup.py build develop

unset INSTALL_DIR

# NOTE: Or just follow soumith's gist: https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b
# conda install -c conda-forge overrides --no-deps
conda uninstall --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -c conda-forge libjpeg-turbo --no-deps
wget https://github.com/uploadcare/pillow-simd/archive/refs/tags/9.0.1.zip
cd pillow-simd-9.0.1/
CC="cc -mavx2" python setup.py install
conda install -c conda-forge --no-deps jpeg libtiff 
pip install overrides tensorboard torchinfo
python -c 'import PIL; print(PIL.__version__)'
python -c 'import PIL.features; print(PIL.features.check_feature("libjpeg_turbo"))'

