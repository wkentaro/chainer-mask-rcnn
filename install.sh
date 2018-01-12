#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup Anaconda {{

CONDA_ACTIVATE_FILE=$HERE/.anaconda2/bin/activate
if [ ! -e $CONDA_ACTIVATE_FILE ]; then
  curl https://raw.githubusercontent.com/wkentaro/dotfiles/bd8bb9bd1a2a440eb6265a2f022f9f74dbda2f1b/local/bin/install_anaconda2.sh | bash -s .
fi
source $CONDA_ACTIVATE_FILE
conda --version
conda info -e

# }}

conda install -q -y -c menpo opencv
pip install -U numpy

conda install -q -y pyqt

# install mvtk
pip install Cython
pip install scikit-image
pip install git+https://github.com/wkentaro/mvtk.git

pip install -e .

set +e
