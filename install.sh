#!/bin/bash

set -e

if [ -e ~/.anaconda2/envs/mask_rcnn ]; then
  source ~/.anaconda2/envs/mask_rcnn/bin/activate
else
  source ~/.anaconda2/bin/activate
  conda create --name mask_rcnn
  source activate mask_rcnn
fi

set -x

pip install Cython

(cd chainer && pip install -e .)

(cd chainercv && pip install -e .)

pip install -e .

set +x

set +e
