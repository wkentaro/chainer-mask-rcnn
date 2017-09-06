#!/usr/bin/env python

import argparse
import datetime

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np

import mask_rcnn

from train import transform_lsvrc2012_vgg16


class FasterRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, instance_dataset):
        self._instance_dataset = instance_dataset

    def __len__(self):
        return len(self._instance_dataset)

    def get_example(self, i):
        from mask_rcnn.utils import label2instance_boxes
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        labels, bboxes = label2instance_boxes(lbl_ins, lbl_cls)
        labels = labels.astype(np.int32, copy=False)
        bboxes = bboxes[:, [1, 0, 3, 2]]  # xy -> yx
        bboxes = bboxes.astype(np.float32, copy=False)
        scale = np.array(1, dtype=np.float32)
        return img, bboxes, labels, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, required=True)
    args = parser.parse_args()

    gpu = args.gpu

    faster_rcnn_model = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(
        n_fg_class=20, pretrained_model='imagenet')
    model = mask_rcnn.models.faster_rcnn.FasterRCNNTrainChain(
        faster_rcnn_model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=1e-3, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

    dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset = chainer.datasets.TransformDataset(
        dataset, transform_lsvrc2012_vgg16)
    dataset = FasterRcnnDataset(dataset)
    iter_train = chainer.iterators.SerialIterator(dataset, batch_size=1)

    out = 'logs/faster_rcnn_cfg001_%s' % \
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    updater = training.StandardUpdater(iter_train, optimizer, device=gpu)
    trainer = training.Trainer(updater, (100, 'epoch'), out=out)

    trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1),
                   trigger=(50000, 'epoch'))

    trainer.extend(
        extensions.snapshot_object(
            model.faster_rcnn,
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
