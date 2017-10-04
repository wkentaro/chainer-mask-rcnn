#!/bin/bash

source ~/.anaconda2/envs/mask_rcnn/bin/activate || \
  conda create --name mask_rcnn && \
  source activate mask_rcnn

set -x

pip install Cython

(cd chainer && pip install -e .)

(cd chainercv && pip install -e .)

pip install -e .

set +x
