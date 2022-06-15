conda create -n sgb python=3.8
conda activate sgb
# CUDA 11.4 Not working for Pytorch 1.11.0. Try 1.10.1.
# The only way 1.10.1 works is if you install it first before anything else.
# Verify with import torch; torch.cuda.is_available(). Should be true.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install ipython ninja yacs cython matplotlib tqdm
# TODO: build h5py with parallel support

pip install opencv-python setuptools==59.5.0 scipy

mkdir ~/sgb && cd ~/sgb
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
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# TODO: Edit out the raise RuntimeError in setup.py if there's a minor version mismatch.

# install PyTorch Detection
cd $INSTALL_DIR
git clone git@github.com:zhanwenchen/SGG-G2S.git gsc
cd gsc

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR

# conda install -c conda-forge overrides tensorboard --no-deps
conda uninstall --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -c conda-forge libjpeg-turbo --no-deps
wget https://github.com/uploadcare/pillow-simd/archive/refs/tags/9.0.1.zip
cd pillow-simd-9.0.1/
CC="cc -mavx2" python setup.py install
conda install -c conda-forge --no-deps jpeg libtiff overrides tensorboard
python -c 'import PIL; print(PIL.__version__)'
python -c 'import PIL.features; print(PIL.features.check_feature("libjpeg_turbo"))'
