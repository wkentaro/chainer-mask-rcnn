#!/usr/bin/env python

import argparse
import datetime

import chainer
from chainer import training
from chainer.training import extensions
import fcn

import mask_rcnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, required=True)
    args = parser.parse_args()

    gpu = args.gpu

    mask_rcnn_model = mask_rcnn.models.MaskRcnn()
    model = mask_rcnn.models.MaskRcnnTrainChain(mask_rcnn_model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=0.02, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset = chainer.datasets.TransformDataset(
        dataset, fcn.datasets.transform_lsvrc2012_vgg16)
    dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset)
    iter_train = chainer.iterators.SerialIterator(dataset, batch_size=1)

    out = 'logs/%s' % datetime.datetime.now().strftime('cfg001_%Y%m%d_%H%M%S')

    updater = training.StandardUpdater(iter_train, optimizer, device=gpu)
    trainer = training.Trainer(updater, (100, 'epoch'), out=out)

    trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.97),
                   trigger=(1, 'epoch'))

    trainer.extend(
        extensions.snapshot_object(
            model, filename='model_snapshot_iter_{.updater.iteration:08}.npz'),
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
