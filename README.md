# Installing FFB6D in Cuda 11.3
The original FFB6D codes were built on Cuda 10.x and opencv 3.x. System with more recent RTX cards can only run with Cuda 11.x

Cuda 10.2: Support Compute capability <= 7.5 <br/>
Cuda 11.0: Support Compute capability <= 8.0 <br/>
Cuda 11.3: Support Compute capability <= 8.6 <br/>

GPU cards compute capability: <br/>
RTX 3090, 3080, 3070: Compute capability 8.6 <br/>
RTX 2080 Ti: Compute capability 7.5 <br/>
GTX 1080 Ti: Conpute capability 6.1 <br/>

Our new server is already installed with Cuda 11.3 but this is to record the step to install cuda 11.3 on the system.
You can skip these instructions for installing cuda 11.3 for the system
## Installing Cuda 11.3 and nvidia display drivers
```
sudo apt purge -y 'nvidia-*'
sudo apt autoremove
```
Installing nvidia driver:
```
sudo apt install nvidia-driver-470 #version 470
```

## Installing FFB6D

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
tk
tqdm
```

```
pip install -r requirement_new.txt
```
## Installing Opencv 3.x
OpenCV need to be compilted as this is required latter when other packages require compiling with the opencv header. <br/>
The "make install" will install the headers and libraries to the conda system and also the python cv2 package 
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
## Install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
python setup.py install --cuda_ext --cpp_ext
cd ..

```
## Install normal speed
Install normalSpeed, a fast and light-weight normal map estimator. This will require opencv headers:
```
git clone https://github.com/hfutcgncas/normalSpeed.git
cd normalSpeed/normalSpeed
python3 setup.py install --user
cd ..
```
## Compile RandLA-Net operators:
```
cd ffb6d/models/RandLA/
sh compile_op.sh
```
## FFB6D codes update
The original codes cant run after the first epouch because of tensorboardx problem.
Please replace line 418 in the file ffb6d/train_lm.py (writer.add_scalars('val_acc', acc_dict, it)) with:

```
          for i,val in enumerate(acc_dict):
                writer.add_scalar(tag='Checking range', scalar_value=acc_dict[val][i], global_step=i)
```

## Training with LineMod

### Generate rendered and Fused data for LineMod datasets
The raster triangle codes has been downloaded to /home/datasets/raster_triangle. Compile the rastertriangle_so.sh by modifying the correct library and include directory

Generate the  rendered and fused data by:
E.g for 'ape'
```
python3 rgbd_renderer.py --cls ape --render_num 70000
python3 fuse.py --cls ape --fuse_num 10000
```
Note there are following objects to generate and train:
```
    'ape': 1,
    'benchvise': 2,
    'cam': 4,
    'can': 5,
    'cat': 6,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15,
```
Finally, training using the following commands:

```
cd ffb6d
n_gpu=4
cls='ape'
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls
```
Make sure there is no warning that apex is not compiled with --cuda_ext option, otherwise pip uninstall apex and compile apex again
## Vscode debugging

To debug on vscode, set the launch.json as follows:

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.launch",
            "console": "integratedTerminal",
            "args": ["--nproc_per_node=2", "train_lm.py", "--gpus=2", "--cls=ape"]
        }
    ]
}
```
or
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "/home/kplim/miniconda3/envs/ffb6d/lib/python3.7/site-packages/torch/distributed/launch.py",
            "cwd" : "path to script"
            "console": "integratedTerminal",
            "args": ["--nproc_per_node=4", "train_lm.py", "--gpus=4", "--cls=ape"]
        }
    ]
}
```
# Codes Run Through

## Inputs to train model

```
dpt_map_m=dpt_m : [480,640] depth map (already divided by scale of 1000, i.e. in mm)

```

xyz_lst is a list of pointclouds in the scene. Each higher index is the down-sampled version of original <br/>
Note xyz_lst[0] is not sampled point cloud

```
xyz_lst is a 2D RGB locations containing XYZ coordinates at each rgb location

xyz_lst[0]-> [3,480,640]
xyz_lst[1]-> [3,240,320]
xyz_lst[2]-> [3,120,160]
xyz_lst[3]-> [3,60,80]

sr2dptxyz is a tuple and is linearised version of xyz_lst:
sr2dptxyz[1] : linear array of size [480*460 , 3]
sr2dptxyz[2] : linear array of size [240*320 , 3]
sr2dptxyz[4] : linear array of size [120*160 , 3]
sr2dptxyz[8] : linear array of size [60*80 , 3]
```

```
cld ???[n_sample_points,3] Chosen number of sample points (n_sample_points) from the actual point cloud of the scene
       Note also that the consecutive position in the linear array does not mean they are closest point. It has been randomised by the index choose
rgb_pt: [n_sample_points,3] The corresponding rgb values in cld
nrm_pt: [n_sample_points,3] The corresponding depth normals in cld
labels=labels_pt:  [n_sample_points] The mask of the object
choose: [n_sample_points] Index of chosen points
cld_rgb_nrm : [9, n_sample_points] concatenate cld,rgb,nrm at each point 
```
```
RTs : only [0], pose (R*T) of object e.g. 'APE' 
kp_3ds=kp3ds : 8 selected keypts in APE multiplied by RT i.e. the 3d keypoints locations (XYZ) at the scene
ctr_3ds=ctr3ds : the pose of the centre of the 3d object mulitplied by RT i.e. the 3d location (XYZ) of the centre of the object at the scene
cls_ids : class id
kp_targ_ofst: object ptcld centered at each keypoint (each keypoint is zero coordinate). 
              #Computed by minusing obj ptcld with each kp3ds
ctr_targ_ofst: the pointcloud (from cld) of the object center as zero coordinate  and masking background out (left only object ptcld)
```
```
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """
```


```
cld_xyz_ (_ = 0 to 3):
cld_xyz0 is original cld. [bs, 12800, 3]
cld_xyz1 is sub sampling cld_xyz0 by 4. [bs, 3200, 3]
cld_xyz2 is sub sampling cld_xyz1 by 4 [bs, 800, 3]
cld_xyz3 is is sub sampling cld_xyz2 by 4 [bs, 200, 3]

cld_nei_idx_ 
contain 16 indices of the neighbour of each point on cld_xyz_. neighbours are the points in cld_xyz_
cld_nei_idx0 : [bs, 12800, 16]
cld_nei_idx1 : [bs, 3200, 16]
cld_nei_idx2 : [bs, 800, 16]
cld_nei_idx3 : [bs, 200, 16]

For explanation purpose, let sub_pts_ be
sub_pts0 [bs, 3200, 3] be the point cloud of subsampled of cld_xyz0 by 4
sub_pts1 [bs, 800, 3] be the point cloud of subsampled of cld_xyz1 by 4
sub_pts2 [bs, 200, 3] be the point cloud of subsampled of cld_xyz2 by 4
sub_pts3 [bs, 50, 3] be the point cloud of subsampled of cld_xyz3 by 4

cld_sub_idx_:
contain indices of 16 neighbours of sub_pts_
cld_sub_idx0: [bs, 3200, 16], the indices of 16 nearest neightbours of sub_pts0
cld_sub_idx1: [bs, 800, 16], the indices of 16 nearest neightbours of sub_pts1
cld_sub_idx2: [bs, 200, 16], the indices of 16 nearest neightbours of sub_pts2
cld_sub_idx3: [bs, 50, 16], the indices of 16 nearest neightbours of sub_pts3


cld_interp_idx_:
contain index of the nearest neighbour of each point on cld_xyz_. The nearest neighbour is from sub_pts_
cld_interp_idx0: [bs, 12800, 1] contain index of the nearest neighbour of each point on cld_xyz0. Nearest neighbour is from sub_pts0 [bs, 3200, 3]
cld_interp_idx1: [bs, 3200, 1] contain index of the nearest neighbour of each point on cld_xyz1. Nearest neighbour is from sub_pts1 [bs, 800, 3]
cld_interp_idx2: [bs, 800, 1] contain index of the nearest neighbour of each point on cld_xyz2. Nearest neighbour is from sub_pts2 [bs, 200, 3]
cld_interp_idx3: [bs, 200, 1] contain index of the nearest neighbour of each point on cld_xyz3. Nearest neighbour is from sub_pts3 [bs, 50, 3]
```
```
sr2dptxyz is the original 2D rgb locations linearised that contain XYZ coordinate for each rgb value
sr2dptxyz[1]: (307200, 3) The original 2D linearised rgb locations with XYZ coordinate. Note: 480*640/1=307200
sr2dptxyz[2]: (76800 , 3) The original 2D linearised rgb locations with XYZ coordinate sub sampled by 4. Note: 480*640/4=76800
sr2dptxyz[4]: (19200 , 3) The original 2D linearised rgb locations with XYZ coordinate sub sampled by 16. Note: 480*640/16=19200
sr2dptxyz[8]: (4800 , 3) The original 2D linearised rgb locations with XYZ coordinate sub sampled by 64. Note: 480*640/64=4800


r2p_ds_nei_idx0: [bs, 3200, 16] contain 16 indices of the neighbour of each point on sub_pts0. neighbours are from sr2dptxyz[4] (19200 , 3) 
r2p_ds_nei_idx1: [bs, 800, 16] contain 16 indices of the neighbour of each point on sub_pts1. neighbours are from sr2dptxyz[8] (4800 , 3)
r2p_ds_n1i_idx2: [bs, 200, 16] contain 16 indices of the neighbour of each point on sub_pts2. neighbours are from sr2dptxyz[8] (4800 , 3)
r2p_ds_n2i_idx3: [bs, 50, 16] contain 16 indices of the neighbour of each point on sub_pts3. neighbours are from sr2dptxyz[8] (4800 , 3)
```

Top to bottom (4 layers):
```
p2r_ds_nei_idx0: [bs, 19200, 1] Each rgb location of sr2dptxyz['4'] store the index of the nearest point in the pt cloud sub_pts0 [bs, 3200, 3]
p2r_ds_nei_idx1: [bs, 4800, 1] Each rgb location of sr2dptxyz['8'] store the index of the nearest point in the pt cloud sub_pts0 [bs, 800, 3]
p2r_ds_nei_idx2: [bs, 4800, 1] Each rgb location of sr2dptxyz['8'] store the index of the nearest point in the pt cloud sub_pts0 [bs, 200, 3]
p2r_ds_nei_idx3: [bs, 4800, 1] Each rgb location of sr2dptxyz['8'] store the index of the nearest point in the pt cloud sub_pts0 [bs, 50, 3]
```

Bottom to Top (3 layers):
```
r2p_up_nei_idx0: contain 16 indices of the neighbour of each point on cld_xyz3. neighbours are from sr2dptxyz[4]
r2p_up_nei_idx1: contain 16 indices of the neighbour of each point on cld_xyz2. neighbours are from sr2dptxyz[2]
r2p_up_nei_idx2: contain 16 indices of the neighbour of each point on cld_xyz1. neighbours are from sr2dptxyz[2]

p2r_up_nei_idx0: contain index of 1 nearest neighbour of each point on sr2dptxyz[4]. neighbours are from cld_xyz3
p2r_up_nei_idx1: contain index of 1 nearest neighbour of each point on sr2dptxyz[2]. neighbours are from cld_xyz2
p2r_up_nei_idx2: contain index of 1 nearest neighbour of each point on sr2dptxyz[2]. neighbours are from cld_xyz1
``` 
```
relative_pos_encoding
[relative_dis, relative_xyz, xyz_tile, neighbor_xyz]
return [bs, 12800, 16, 10]

xyz_tile: [bs, 12800, 16, 3], cordinate of each pt in the point cloud repeated 16 times
neighbor_xyz: [bs, 12800, 16, 3], the 16 cordinate of the neigbout of each pt in the point cloud. each coordinate is (x,y,z)
relative_xyz:  [bs, 12800, 16, 3], xyz_tile-neighbor_xyz which is the vector from each neighbour to the point in the pointcloud
relative_dis: [bs, 12800, 16, 1], the length of each vector in relative_xyz

```
