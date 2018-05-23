import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.4.1'


if sys.argv[-1] == 'release':
    commands = [
        'git tag v{0}'.format(version),
        'git push origin master --tags',
        'python setup.py sdist upload',
    ]
    for cmd in commands:
        subprocess.call(cmd, shell=True)
    sys.exit(0)


setup(
    name='chainer-mask-rcnn',
    version=version,
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
