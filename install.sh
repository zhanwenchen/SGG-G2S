conda create -n sgb python=3.8
conda activate sgb
# CUDA 11.4 Not working for Pytorch 1.11.0. Try 1.10.1.
# The only way 1.10.1 works is if you install it first before anything else.
# Verify with import torch; torch.cuda.is_available(). Should be true.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge overrides
conda install ipython scipy h5py ninja yacs cython matplotlib tqdm
pip install opencv-python

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
# python setup.py install --cuda_ext --cpp_ext
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# TODO: Edit out the raise RuntimeError in setup.py if there's a minor version mismatch.

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git scene-graph-benchmark
cd scene-graph-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
