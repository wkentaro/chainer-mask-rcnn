#!/bin/bash

set -e
set -x

if [ ! -e venv ]; then
  unset PYTHONPATH
  virtualenv venv
fi

set +x
. venv/bin/activate
set -x

cp $(/usr/bin/python -c 'import cv2; print(cv2.__file__)') venv/lib/python2.7/site-packages
pip install -U numpy

cp -r $(dirname $(/usr/bin/python -c 'import PyQt4; print(PyQt4.__file__)')) venv/lib/python2.7/site-packages
cp $(/usr/bin/python -c 'import sip; print(sip.__file__)') venv/lib/python2.7/site-packages

pip install Cython

pip install git+https://github.com/wkentaro/mvtk.git

(cd chainer && pip install -e .)

(cd chainercv && pip install -e .)

pip install -e .

set +x
set +e
