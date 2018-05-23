# chainer-mask-rcnn

[![PyPI version](https://badge.fury.io/py/chainer-mask-rcnn.svg)](https://badge.fury.io/py/chainer-mask-rcnn)
[![Build Status](https://travis-ci.com/wkentaro/chainer-mask-rcnn.svg?branch=master)](https://travis-ci.com/wkentaro/chainer-mask-rcnn)

Chainer Implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).  
It includes:

- ResNet50, ResNet101 backbone (FPN not included)
- [VOC and COCO training examples](examples).

<img src="examples/coco/.readme/R-50-C4_x1_33823288584_1d21cf0a26_k.jpg" width="44.3%" /> <img src="examples/coco/.readme/R-50-C4_x1_17790319373_bd19b24cfc_k.jpg" width="52%" />  
*Fig 1. Mask R-CNN, ResNet50, 8GPU, COCO 31.4 mAP@50:95*


## COCO Results

| Model | Implementation | N gpu training | mAP@50:95 |
|-------|----------------|----------------|-----------|
| Mask R-CNN, ResNet50 | [Ours](https://github.com/wkentaro/chainer-mask-rcnn) | 8 | 31.4 |
| Mask R-CNN, ResNet50 | [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron) | 8 | 31.4 (30.7 after copied) |
| FCIS, ResNet50 | [msracver/FCIS](https://github.com/msracver/FCIS) | 8 | 27.1 |


## Training

See [examples](examples).


## Inference

```bash
# you can use your trained model
./demo.py logs/<YOUR_TRAINING_LOG> --img <IMAGE_PATH_OR_URL>

# copy weight from caffe2 to chainer
cd examples/coco
./convert_caffe2_to_chainer.py
./demo.py logs/R-50-C4_x1_caffe2_to_chainer --img https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/33823288584_1d21cf0a26_k.jpg
./demo.py logs/R-50-C4_x1_caffe2_to_chainer --img https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/17790319373_bd19b24cfc_k.jpg
```

<img src="examples/coco/.readme/R-50-C4_x1_caffe2_to_chainer_result_33823288584_1d21cf0a26_k.jpg" width="44.3%" /> <img src="examples/coco/.readme/R-50-C4_x1_caffe2_to_chainer_result_17790319373_bd19b24cfc_k.jpg" width="52%" />  
*Fig 2. Mask R-CNN, ResNet50, 8GPU, Copied from Detectron, COCO 31.4 mAP@50:95*



## Installation


### For single GPU training

```bash
pip install opencv-python
pip install Cython
pip install .
```


### For multi GPU training

- Install [OpenMPI](https://www.open-mpi.org/software/ompi/v3.0/)

```bash
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
tar zxvf openmpi-3.0.0.tar.gz
cd openmpi-3.0.0
./configure --with-cuda
make -j4
sudo make install
sudo ldconfig
```

- Install [NCCL](https://developer.nvidia.com/nccl)

```bash
# dpkg -i nccl-repo-ubuntu1404-2.1.4-ga-cuda8.0_1-1_amd64.deb
dpkg -i nccl-repo-ubuntu1604-2.1.15-ga-cuda9.1_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev
```

- Install [ChainerMN](https://github.com/chainer/chainermn)

```bash
pip install chainermn
```

- Install [chainer-mask-rcnn](https://github.com/wkentaro/chainer-mask-rcnn)

```bash
pip install opencv-python
pip install Cython
pip install .
```


## Testing

```bash
pip install flake8 pytest
flake8 .
pytest -v tests
```


## ROS integration (real-time demo)

It runs in around 10fps with Titan X Pascal.  
See https://github.com/jsk-ros-pkg/jsk_recognition/pull/2257.

<img src="https://user-images.githubusercontent.com/4310419/38289831-39bc0534-3813-11e8-9ab4-d774ab4289b4.gif" width="66%" />
