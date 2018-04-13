#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint

import cv2  # NOQA

import chainer
import numpy as np
import yaml

import chainer_mask_rcnn as mrcnn


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help='Log dir.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    args = parser.parse_args()

    log_dir = args.log_dir

    # param
    params = yaml.load(open(osp.join(log_dir, 'params.yaml')))
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    # dataset
    test_data = mrcnn.datasets.SBDInstanceSegmentationDataset('val')
    fg_class_names = test_data.class_names

    def transform_test_data(in_data):
        img = in_data[0]
        img = img.transpose(2, 0, 1)
        out_data = list(in_data)
        out_data[0] = img
        return tuple(out_data)

    test_data = chainer.datasets.TransformDataset(
        test_data, transform_test_data)

    # model
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    if params['pooling_func'] == 'align':
        pooling_func = mrcnn.functions.roi_align_2d
    elif params['pooling_func'] == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    elif params['pooling_func'] == 'resize':
        pooling_func = mrcnn.functions.crop_and_resize
    else:
        raise ValueError

    model = params['model']
    pretrained_model = osp.join(log_dir, 'snapshot_model.npz')
    mask_rcnn = mrcnn.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(fg_class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        roi_size=params.get('roi_size', 7)
    )
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()

    # visualization
    # -------------------------------------------------------------------------

    test_vis_data = mrcnn.datasets.IndexingDataset(
        test_data, indices=[196, 204, 216, 257, 326, 473, 566, 649, 1063])
    test_vis_iter = chainer.iterators.SerialIterator(
        test_vis_data, batch_size=1, repeat=False, shuffle=False)

    class DummyTrainer(object):

        class DummyUpdater(object):

            iteration = 'best'

        updater = DummyUpdater()
        out = log_dir

    print('Visualizing...')
    visualizer = mrcnn.extensions.InstanceSegmentationVisReport(
        test_vis_iter, mask_rcnn,
        label_names=fg_class_names,
        file_name='iteration=%s.jpg',
        copy_latest=False,
    )
    visualizer(trainer=DummyTrainer())
    print('Saved visualization:', osp.join(log_dir, 'iteration=best.jpg'))

    # evaluation
    # -------------------------------------------------------------------------

    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    print('Evaluating...')
    mask_rcnn.use_preset('evaluate')
    evaluator = mrcnn.extensions.InstanceSegmentationVOCEvaluator(
        test_iter, mask_rcnn, use_07_metric=True,
        label_names=fg_class_names, show_progress=True)
    result = evaluator()

    for k in result:
        if isinstance(result[k], np.float64):
            result[k] = float(result[k])

    yaml_file = pretrained_model + '.eval_result.yaml'
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print('Saved evaluation: %s' % yaml_file)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
