from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError:
    print('Please install Cython.')
    quit(1)

try:
    import numpy as np
except ImportError:
    print('Please install Numpy.')
    quit(1)


ext_modules = [
    Extension(
        'chainer_mask_rcnn.external.pycocotools._mask',
        sources=['chainer_mask_rcnn/external/pycocotools/common/maskApi.c',
                 'chainer_mask_rcnn/external/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(),
                      'chainer_mask_rcnn/external/pycocotools/common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]


setup(
    name='chainer_mask_rcnn',
    version='0.1',
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
    ext_modules=cythonize(ext_modules),
)
