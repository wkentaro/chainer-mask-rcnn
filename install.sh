#!/bin/bash

set -e
set -x

# Setup Anaconda {{

if [ ! -e $HOME/.anaconda2/bin/activate ]; then
  echo 'Please install Anaconda to ~/.anaconda2'
  exit 1
fi
unset PYTHONPATH
set +x && source $HOME/.anaconda2/bin/activate && set -x
conda --version

if [ ! -e $CONDA_PREFIX/envs/mask-rcnn ]; then
  conda create -q -y --name=mask-rcnn python=2.7
fi
set +x && source activate mask-rcnn && set -x

conda info -e

# }}

conda install -q -y -c menpo opencv
conda install -q -y pyqt

pip install Cython

pip install git+https://github.com/wkentaro/mvtk.git

(cd chainer && pip install -e .)

pip install -e .

set +x
set +e
