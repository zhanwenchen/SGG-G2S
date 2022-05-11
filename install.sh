conda create -n sgb python=3.8
conda activate sgb
conda install -c conda-forge overrides

conda install ipython scipy h5py ninja yacs cython matplotlib tqdm opencv
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # does not work with Pytorch 1.11.0
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch

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

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
