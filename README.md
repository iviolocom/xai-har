

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
