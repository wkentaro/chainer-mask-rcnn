# VOC Example


## Usage

```bash
./download_datasets.py

# single gpu training
./train.py --gpu 0

# multi gpu training
mpiexec -n 4 ./train.py --multi-node
```


## Result

| Model                          | mAP  |
|--------------------------------|------|
| Mask R-CNN, ResNet50-C4 [Ours] | 65.3 |
| FCIS, ResNet-v1-101, [[msracver/FCIS]](https://github.com/msracver/FCIS) | 66.0 |
