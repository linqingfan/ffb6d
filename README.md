# Installing ffb6d in Cuda 11.3

If cuda 11.3 is not installed yet, uninstal nvidia drivers first:
```
sudo apt purge -y 'nvidia-*'

```
Installing nvidia driver:
```
sudo apt install nvidia-driver-470 #version 470
```
```
git clone https://github.com/ethnhe/FFB6D
cd FFB6D
conda create -n ffb6d python=3.7
conda activate ffb6d
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Create a file requirement_new.txt with the following content:
```
h5py
numpy
#torch==1.4.0
#torchvision==0.2.0
PyYAML==5.4.1
#pprint
enum34
future
Cython
cffi
scipy==1.4.1
pybind11[global]
#opencv_python==3.4.8.29
#opencv_contrib_python==3.4.8.29
#matplotlib==3.0.2
matplotlib
transforms3d==0.3.1
scikit_image
lmdb==0.94
setuptools==41.0.0
#cffi==1.11.5
easydict==1.7
plyfile==0.6
#glumpy==1.0.6
glumpy
pillow==8.2.0
tensorboardx
pandas
sklearn
termcolor
```

```
pip install -r requirement_new.txt
```

```
wget https://github.com/opencv/opencv/archive/3.4.16.zip
unzip 3.4.16.zip
rm 3.4.16.zip
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.16.zip
unzip 3.4.16.zip
rm 3.4.16.zip
cd opencv-3.4.16
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX -D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/python3.7 -D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.7m -D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python -D PYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.7/site-packages -D WITH_TBB=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_CUDA=ON -D BUILD_opencv_cudacodec=OFF -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.16/modules -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..

make -j8
make install
```

```
git clone https://github.com/NVIDIA/apex
cd apex
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" ./
cd ..

```

Install normalSpeed, a fast and light-weight normal map estimator:
```
git clone https://github.com/hfutcgncas/normalSpeed.git
cd normalSpeed/normalSpeed
python3 setup.py install --user
cd ..
```
```
for i,val in enumerate(acc_dict):
                writer.add_scalar(tag='Checking range', scalar_value=acc_dict[val][i], global_step=i)
```
