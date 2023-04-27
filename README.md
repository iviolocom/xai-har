Estimate each performance of each model on the actions
```
nohup python inference_each_set_video_dashboard.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logdb0412.out&
```
```
nohup python inference_each_set_video_rearview.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logrv0412.out&
```
```
nohup python inference_each_set_video_right.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR datasets/ > logs/logrs0412.out&
```
