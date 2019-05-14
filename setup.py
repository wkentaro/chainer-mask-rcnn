from __future__ import print_function

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.5.21'


if sys.argv[-1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/chainer-mask-rcnn-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()

    try:
        import github2pypi
    except ImportError:
        return long_description

    return github2pypi.replace_url(
        slug='wkentaro/chainer-mask-rcnn', content=long_description
    )


setup(
    name='chainer-mask-rcnn',
    version=version,
    packages=find_packages(exclude=['github2pypi']),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    description='Chainer Implementation of Mask R-CNN.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='http://github.com/wkentaro/chainer-mask-rcnn',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
