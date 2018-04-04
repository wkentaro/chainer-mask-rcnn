# COCO Example

## Usage

```bash
./download_datasets.py

# single gpu training
./train.py --gpu 0

# multi gpu training
mpiexec -n 4 ./train.py --multi-node

./evaluate.py logs/<log_dir> --gpu 0
./demo.py logs/<log_dir> --gpu 0
./demo.py logs/<log_dir> --gpu 0 --img https://github.com/facebookresearch/Detectron/blob/master/demo/17790319373_bd19b24cfc_k.jpg?raw=true
```


## Result

| Model | Implementation | N gpu training | mAP@50:95 |
|-------|----------------|----------------|-----------|
| Mask R-CNN, ResNet50 | [Ours](https://github.com/wkentaro/chainer-mask-rcnn) | 4 | 28.3 |
| Mask R-CNN, ResNet50 | [Ours](https://github.com/wkentaro/chainer-mask-rcnn) | 8 | |
| Mask R-CNN, ResNet50 | [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron) | 8 | **31.4** |
| FCIS, ResNet50 | [msracver/FCIS](https://github.com/msracver/FCIS) | 8 | 27.1 |
