# chainer-mask-rcnn

[![PyPI version](https://badge.fury.io/py/chainer-mask-rcnn.svg)](https://badge.fury.io/py/chainer-mask-rcnn)
[![Python Versions](https://img.shields.io/pypi/pyversions/chainer-mask-rcnn.svg)](https://pypi.org/project/chainer-mask-rcnn)
[![Build Status](https://travis-ci.com/wkentaro/chainer-mask-rcnn.svg?branch=master)](https://travis-ci.com/wkentaro/chainer-mask-rcnn)

Chainer Implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).

## Features

- [x] ResNet50, ResNet101 backbone.
- [x] [VOC and COCO training examples](examples).
- [x] **[Reproduced result of original work (ResNet50, COCO)](#coco-results)**.
- [x] Weight copy from pretrained model at [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron).
- [x] Training with batch size >= 2.
- [ ] Support FPN backbones.
- [ ] Keypoint detection.

<img src="examples/coco/.readme/R-50-C4_x1_33823288584_1d21cf0a26_k.jpg" width="44.3%" /> <img src="examples/coco/.readme/R-50-C4_x1_17790319373_bd19b24cfc_k.jpg" width="52%" />  
*Fig 1. Mask R-CNN, ResNet50, 8GPU, Ours, COCO 31.4 mAP@50:95*



## COCO Results

| Model | Implementation | N gpu training | mAP@50:95 | Log |
|-------|----------------|----------------|-----------|-----|
| Mask R-CNN, ResNet50 | [Ours](.) | 8 | 31.5 - 31.8 | [Log](https://drive.google.com/open?id=1WOEtVnxqYdHl35pAyIcp-H0HtTjI-l3V) |
| Mask R-CNN, ResNet50 | [Detectron](https://github.com/facebookresearch/Detectron) | 8 | 31.4 (30.8 after copied) | [Log](https://drive.google.com/open?id=1xQBox3uMv2FoyXXpsC9ASNZ-92NgAbcT) |
| FCIS, ResNet50 | [FCIS](https://github.com/msracver/FCIS) | 8 | 27.1 | - |


## Inference

```bash
# you can use your trained model
./demo.py logs/<YOUR_TRAINING_LOG> --img <IMAGE_PATH_OR_URL>

# COCO Example: Mask R-CNN, ResNet50, 31.4 mAP@50:95
cd examples/coco
LOG_DIR=logs/20180730_081433
mkdir -p $LOG_DIR
pip install gdown
gdown https://drive.google.com/uc?id=1XC-Mx4HX0YBIy0Fbp59EjJFOF7a3XK0R -O $LOG_DIR/snapshot_model.npz
gdown https://drive.google.com/uc?id=1fXHanL2pBakbkv83wn69QhI6nM6KjrzL -O $LOG_DIR/params.yaml
./demo.py $LOG_DIR

# copy weight from caffe2 to chainer
cd examples/coco
./convert_caffe2_to_chainer.py  # or download from https://drive.google.com/open?id=1WOEtVnxqYdHl35pAyIcp-H0HtTjI-l3V
./demo.py logs/R-50-C4_x1_caffe2_to_chainer --img https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/33823288584_1d21cf0a26_k.jpg
./demo.py logs/R-50-C4_x1_caffe2_to_chainer --img https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/17790319373_bd19b24cfc_k.jpg
```

<img src="examples/coco/.readme/R-50-C4_x1_caffe2_to_chainer_result_33823288584_1d21cf0a26_k.jpg" width="44.3%" /> <img src="examples/coco/.readme/R-50-C4_x1_caffe2_to_chainer_result_17790319373_bd19b24cfc_k.jpg" width="52%" />  
*Fig 2. Mask R-CNN, ResNet50, 8GPU, Copied from Detectron, COCO 31.4 mAP@50:95*


## Installation & Training


### Single GPU Training

```bash
# Install Chainer Mask R-CNN.
pip install opencv-python
pip install .

# Run training!
cd examples/coco && ./train.py --gpu 0
```


### Multi GPU Training

```bash
# Install OpenMPI
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
tar zxvf openmpi-3.0.0.tar.gz
cd openmpi-3.0.0
./configure --with-cuda
make -j4
sudo make install
sudo ldconfig

# Install NCCL
# dpkg -i nccl-repo-ubuntu1404-2.1.4-ga-cuda8.0_1-1_amd64.deb
dpkg -i nccl-repo-ubuntu1604-2.1.15-ga-cuda9.1_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev

# Install ChainerMN
pip install chainermn

# Finally, install Chainer Mask R-CNN.
pip install opencv-python
pip install .

# Run training!
cd examples/coco && mpirun -n 4 ./train.py --multi-node
```


## Testing

```bash
pip install flake8 pytest
flake8 .
pytest -v tests
```
