from setuptools import find_packages
from setuptools import setup


setup(
    name='mask_rcnn',
    version='0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    description='Chainer Implementation of Mask R-CNN.',
    url='http://github.com/wkentaro/mask-rcnn',
    license='MIT',
)
