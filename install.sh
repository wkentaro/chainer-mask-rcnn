#!/bin/bash

set -e
set -x

if [ ! -e ~/.anaconda2/envs/mask_rcnn ]; then
  set +x
  source ~/.anaconda2/bin/activate
  set -x

  conda create -q -y --name=mask_rcnn python=2.7
fi

set +x
source ~/.anaconda2/bin/activate mask_rcnn
set -x

conda info -e

pip install Cython

(cd chainer && pip install -e .)

(cd chainercv && pip install -e .)

pip install -e .

set +x
set +e
