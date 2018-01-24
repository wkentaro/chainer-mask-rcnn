from distutils.extension import Extension

import numpy as np
from setuptools import find_packages
from setuptools import setup


ext_modules = [
    Extension(
        'mvtk.external.pycocotools._mask',
        sources=['mvtk/external/pycocotools/common/maskApi.c',
                 'mvtk/external/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), 'mvtk/external/pycocotools/common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]


setup(
    name='chainer_mask_rcnn',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'chainer_mask_rcnn.datasets.voc': ['data/*'],
    },
    install_requires=open('requirements.txt').readlines(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    description='Chainer Implementation of Mask R-CNN.',
    url='http://github.com/wkentaro/chainer-mask-rcnn',
    license='MIT',
)
