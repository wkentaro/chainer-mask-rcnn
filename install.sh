#!/bin/bash

set -e
set -x

# Initialize
git submodule update --init --recursive

# Setup Anaconda {{

export CONDA_PREFIX=$HOME/.anaconda2

if [ ! -e $CONDA_PREFIX/bin/activate ]; then
  echo 'Please install Anaconda to ~/.anaconda2'
  exit 1
fi
unset PYTHONPATH
set +x && source $CONDA_PREFIX/bin/activate && set -x
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
pip install -U numpy

# install mvtk {
pip install scikit-image
pip install imgaug
pip install 'chainer>=2,<3'
pip install git+https://github.com/wkentaro/mvtk.git
# }

git_branch () {
  set +x
  git branch | grep -e '^*' | awk '{print $2}'
  set -x
}

# install chainercv {
if [ -d chainercv ]; then
  (cd chainercv && test "$(git_branch)" = master)
else
  (git clone https://github.com/chainer/chainercv.git)
fi
(cd chainercv && pip install -e .)
# }

pip install -e .

set +x
set +e
