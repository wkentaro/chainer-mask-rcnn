#!/usr/bin/env python

import argparse
import datetime

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np

import mask_rcnn


def transform_lsvrc2012_vgg16(inputs):
    img = inputs[0]

    # LSVRC2012 used by VGG16
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

    img = img.astype(np.float32)
    img -= MEAN_BGR[::-1]
    img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

    transformed = list(inputs)
    transformed[0] = img
    return tuple(transformed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, required=True)
    args = parser.parse_args()

    gpu = args.gpu

    mask_rcnn_model = mask_rcnn.models.MaskRcnn(pretrained_model='imagenet')
    model = mask_rcnn.models.MaskRcnnTrainChain(mask_rcnn_model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=1e-5, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

    dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset = chainer.datasets.TransformDataset(
        dataset, transform_lsvrc2012_vgg16)
    dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset)
    iter_train = chainer.iterators.SerialIterator(dataset, batch_size=1)

    out = 'logs/%s' % datetime.datetime.now().strftime('cfg001_%Y%m%d_%H%M%S')

    updater = training.StandardUpdater(iter_train, optimizer, device=gpu)
    trainer = training.Trainer(updater, (100, 'epoch'), out=out)

    trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                   trigger=(50000, 'epoch'))

    trainer.extend(
        extensions.snapshot_object(
            model.mask_rcnn,
            filename='model_snapshot_iter_{.updater.iteration:08}.npz'),
        trigger=(5, 'epoch'))

    trainer.extend(
        extensions.LogReport(trigger=(20, 'iteration'), log_name='log.json'))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport([
                'main/loss', 'main/rpn_loc_loss', 'main/rpn_cls_loss',
                'main/roi_loc_loss', 'main/roi_cls_loss',
                'main/roi_mask_loss'
            ], 'epoch', file_name='loss.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'main/loss', 'main/rpn_loc_loss', 'main/rpn_cls_loss',
         'main/roi_loc_loss', 'main/roi_cls_loss',
         'main/roi_mask_loss']))

    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
