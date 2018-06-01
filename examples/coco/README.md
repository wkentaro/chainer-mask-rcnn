# COCO Example

## Usage

```bash
./download_datasets.py

# single gpu training
./train.py --gpu 0

# multi gpu training
mpirun -n 4 ./train.py --multi-node

./evaluate.py logs/<log_dir> --gpu 0
./demo.py logs/<log_dir> --gpu 0 --img <img_file_or_url>
```


## Result

| Model | Implementation | N gpu training | mAP@50:95 | Log |
|-------|----------------|----------------|-----------|-----|
| Mask R-CNN, ResNet50 | [Ours](https://github.com/wkentaro/chainer-mask-rcnn) | 8 | 31.4 | [Log](https://drive.google.com/open?id=1DRt96S0asaAiN5aDiWUYVcPs0bQ_USU3) |
| Mask R-CNN, ResNet50 | [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron) | 8 | 31.4 (30.7 after copied) | [Log](https://drive.google.com/open?id=1xQBox3uMv2FoyXXpsC9ASNZ-92NgAbcT) |
| FCIS, ResNet50 | [msracver/FCIS](https://github.com/msracver/FCIS) | 8 | 27.1 | - |

See [here](https://drive.google.com/open?id=1Dfpc2Dd7_hh9ZsgfbDnuVG4xUnQFBksa) for training logs.

<img src=".readme/resnet50_results.jpg" width="48%" />

*Fig 1. Mask R-CNN, ResNet50, Ours, 31.4 mAP@50:95.*


## Caffe2 (Detectron) to Chainer

```bash
./convert_caffe2_to_chainer.py

./evaluate.py logs/R-50-C4_x1_caffe2_to_chainer

./demo.py logs/R-50-C4_x1_caffe2_to_chainer
```

<img src=".readme/R-50-C4_x1_caffe2_result_33823288584_1d21cf0a26_k.jpg" width="48%" /> <img src=".readme/R-50-C4_x1_caffe2_to_chainer_result_33823288584_1d21cf0a26_k.jpg" width="48%" />  
*Fig 1. Inference results: Caffe2 (left), Chainer (right).*
