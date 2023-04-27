

# Installation

## Requirements
- Python >= 3.8
- Numpy
- PyTorch >= 1.3
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`
- [Detectron2](https://github.com/facebookresearch/detectron2):
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
```
    pip install -U torch torchvision cython
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo
    pip install -e detectron2_repo
    # You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

## Pytorch
Please follow PyTorch official instructions to install from source:
```
git clone --recursive https://github.com/pytorch/pytorch
```

## PySlowFast

Clone the PySlowFast Video Understanding repository.
```
git clone https://github.com/facebookresearch/slowfast
```

Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```

### Build PySlowFast

After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/slowfast
cd SlowFast
python setup.py build develop
```
## 2: Estimate each performance of each model on the actions on independent dataset
- [Download this link](https://mysnu-my.sharepoint.com/:f:/g/personal/haithienld_seoul_ac_kr/Ev2WNnT1p0ZLnQVSnbACi1sB9_Edr8d_3_YXsWVgq39xoA?e=bre30C
)
```
xai-har
  |--- dataset
    |--- *.csv
    |--- folders
```
Runining below command:

```
nohup python inference_each_set_video_dashboard.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logdb0412.out&
```
```
nohup python inference_each_set_video_rearview.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logrv0412.out&
```
```
nohup python inference_each_set_video_right.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logrs0412.out&
```
Then check logs inside the logs/ folder. it created logs file with csv together.

```
,vid_id,action_idxs,action_idxs_counter,action_idxs_len,activities_idx,activities_counter,activities_len,action_idxs_prob,activities_prob
0,0,"[8, 8, 8, 8, 11, 11, 8, 7, 7, 10, 7, 7, 12, 8, 8, 0, 8, 2, 2, 0]",48,147,"[8, 8, 8, 11, 7, 7, 7, 7, 0]",20,61,32.65,32.79
1,1,"[7, 7, 1, 7, 2, 5, 1, 1, 1, 1, 13, 1, 1, 1, 1, 8, 8, 0]",108,170,"[7, 7, 1, 1, 1, 1, 1]",65,74,63.53,87.84
2,2,"[13, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]",125,209,"[2, 2, 3, 3, 3, 3, 3, 3, 3]",75,117,59.81,64.1
3,3,"[3, 3, 3, 3, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3]",118,185,"[3, 3, 3, 3, 3]",75,86,63.78,87.21
4,4,"[7, 1, 7, 7, 7, 7, 7, 13, 7, 1, 7, 13, 7, 7, 13, 7, 13, 8, 7, 4]",40,163,"[1, 7, 13, 13, 13, 7, 13, 7]",22,80,24.54,27.5
5,5,"[9, 13, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 5]",115,188,"[5, 5, 5, 5, 5, 5, 5, 8, 5, 5]",80,106,61.17,75.47
6,6,"[2, 13, 8, 8, 8, 8, 13, 8, 8, 8, 8, 8, 5, 8, 8]",56,189,"[8, 8, 8, 8, 8]",29,81,29.63,35.8
7,7,"[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]",125,158,"[7, 7, 7, 7, 7, 7, 7, 7, 7]",77,86,79.11,89.53
8,8,"[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]",112,191,"[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]",74,103,58.64,71.84
9,9,"[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1]",96,163,"[9, 9, 9, 9, 9, 9, 9, 9]",57,71,58.9,80.28
10,10,"[10, 10, 10, 7, 10, 2, 2, 2, 9, 9, 7, 7, 13, 2]",27,153,"[10, 10, 10, 2, 2, 9]",15,77,17.65,19.48
11,11,"[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12, 11, 12]",12,178,"[12, 12, 12, 12, 12, 12, 12]",1,89,6.74,1.12
12,12,"[7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 7, 12, 12, 12, 7]",78,152,"[7, 7, 7, 7, 12, 12, 12]",41,79,51.32,51.9
13,13,"[13, 13, 13, 13, 7, 14, 7, 7, 7, 7, 7, 1, 13, 1, 1, 13, 1, 13, 13, 1]",74,148,"[13, 13, 7, 7, 7, 7, 13, 1, 1, 1, 1]",43,79,50.0,54.43
14,14,"[7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]",86,174,"[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]",33,80,49.43,41.25
15,15,"[13, 13, 13, 0, 13, 8, 13, 13, 13, 13, 13, 0, 13, 13, 12, 13, 13, 13, 13, 12, 12, 12]",63,186,"[13, 13, 8, 13, 13, 13, 13, 13, 13, 13]",47,88,33.87,53.41

```
