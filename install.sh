conda create -n sgb python=3.8
conda activate sgb
conda install -c conda-forge overrides

conda install ipython scipy h5py ninja yacs cython matplotlib tqdm opencv
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
mkdir ~/sgb && cd ~/sgb
export INSTALL_DIR=$(pwd)
echo $INSTALL_DIR

cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
python setup.py install --cuda_ext --cpp_ext

cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark

python setup.py build develop


unset INSTALL_DIR
