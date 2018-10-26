#!/usr/bin/env python

import argparse
import os.path as osp
import time

import chainer
import chainer_mask_rcnn
import six
import skimage.io
import yaml


def bench_chainer(img_file, gpu, times):
    print('==> Testing Mask R-CNN RestNet50-C4 with Chainer')
    chainer.cuda.get_device(gpu).use()

    chainer.config.train = False
    chainer.config.enable_backprop = False

    log_dir = 'logs/R-50-C4_x1_caffe2_to_chainer'
    with open(osp.join(log_dir, 'params.yaml')) as f:
        params = yaml.load(f)
    pretrained_model = osp.join(log_dir, 'snapshot_model.npz')

    assert params['model'] == 'resnet50'
    assert len(params['class_names']) == 80

    model = chainer_mask_rcnn.models.MaskRCNNResNet(
        n_layers=50,
        n_fg_class=80,
        pretrained_model=pretrained_model,
        anchor_scales=params['anchor_scales'],
        mean=params['mean'],
        min_size=params['min_size'],
        max_size=params['max_size'],
        roi_size=params['roi_size'],
    )
    model.score_thresh = 0.7
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

    img = skimage.io.imread(img_file)
    img_chw = img.transpose(2, 0, 1)

    for i in six.moves.range(5):
        model.predict([img_chw])
    chainer.cuda.Stream().synchronize()
    t_start = time.time()
    for i in six.moves.range(times):
        model.predict([img_chw])
    chainer.cuda.Stream().synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument(
        '--times', type=int, default=100, help='number of times of inference'
    )
    default_img_file = 'https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/33823288584_1d21cf0a26_k.jpg'  # NOQA
    parser.add_argument(
        '--img-file', default=default_img_file, help='image file'
    )
    args = parser.parse_args()

    print('==> Benchmark: gpu=%d, times=%d' % (args.gpu, args.times))
    print('==> Image file: %s' % args.img_file)
    bench_chainer(args.img_file, args.gpu, args.times)


if __name__ == '__main__':
    main()
