# chainer-mask-rcnn

Chainer Implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).


## Installation


### For Single GPU Training

```bash
conda install -c conda-forge -y opencv
pip install .
```


### For Multi GPU Training

- Install [OpenMPI](https://www.open-mpi.org/software/ompi/v3.0/)

```bash
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
tar zxvf openmpi-3.0.0.tar.gz
cd openmpi-3.0.0
./configure --with-cuda
make -j4
sudo make install
sudo ldconfig

pip install chainermn
conda install -c conda-forge -y opencv
pip install .
```


## Testing

```bash
pip install hacking pytest
flake8 .
pytest tests
```
