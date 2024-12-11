# MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps
Created by [Yating Xu](https://pixie8888.github.io/) from National University of Singapore.



## Introduction
This repository contains the PyTorch implementation for NeurIPS 2024 work [MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps](https://arxiv.org/abs/2410.21566). Training and testing are conducted on two A5000 (48GB) GPUs.




## Installation
- Install [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)
- Install packages related to Gaussian Splatting: we install [pixelsplat](https://github.com/dcharatan/pixelsplat).
- Install torch-scatter: ```pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118.html```



## Dataset
### ScanNet
We follow this [instruction](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/NeRF-Det) to prepare ScanNet data.

### ARKitScenes
We follow [CN-RMA](https://github.com/SerCharles/CN-RMA) to prepare ARKitScenes data.


## Train
```
CUDA_VISIBLE_DEVICES=0,1  bash tools/dist_train.sh projects/NeRF-Det/configs/mvsdet_res50_2x_low_res.py 2 --log_dir ModelName
```

## Test
```
CUDA_VISIBLE_DEVICES=0,1  bash tools/dist_test.sh projects/NeRF-Det/configs/mvsdet_res50_2x_low_res.py work_dirs/nerfdet_res50_2x_low_res/Depth\=12_\[0_2_5_0\]_Gaussian+Tgt\=120+MVSPlat+rgb_again/best_mAP_0.25_epoch_11.pth 2
```


## Acknowledgement
We thank [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/NeRF-Det), [PixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSNet](https://github.com/xy-guo/MVSNet_pytorch) for sharing their source code.
