#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint

import chainer
import numpy as np
import yaml

import chainer_mask_rcnn as mrcnn

from train import MaskRCNNDataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    # param
    params = yaml.load(open(osp.join(args.log_dir, 'params.yaml')))
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    # dataset
    test_data = mrcnn.datasets.COCOInstanceSegmentationDataset(
        'minival', use_crowd=True, return_crowd=True, return_area=True)
    class_names = test_data.class_names
    test_data = MaskRCNNDataset(test_data)

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

    min_size = 800
    max_size = 1333
    anchor_scales = (2, 4, 8, 16, 32)

    pretrained_model = osp.join(args.log_dir, 'snapshot_model.npz')
    print('Using pretrained_model: %s' % pretrained_model)

    model = params['model']
    mask_rcnn = mrcnn.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        anchor_scales=anchor_scales,
        min_size=min_size,
        max_size=max_size,
        roi_size=params.get('roi_size', 7)
    )
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()

    test_data = chainer.datasets.TransformDataset(
        test_data, mrcnn.datasets.MaskRCNNTransform(mask_rcnn, train=False))

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
        out = args.log_dir

    print('Visualizing...')
    visualizer = mrcnn.extensions.InstanceSegmentationVisReport(
        test_vis_iter, mask_rcnn,
        label_names=class_names,
        file_name='iteration=%s.jpg',
        copy_latest=False,
    )
    visualizer(trainer=DummyTrainer())
    print('Saved visualization:', osp.join(args.log_dir, 'iteration=best.jpg'))

    # evaluation
    # -------------------------------------------------------------------------

    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    print('Evaluating...')
    mask_rcnn.use_preset('evaluate')
    evaluator = mrcnn.extensions.InstanceSegmentationCOCOEvaluator(
        test_iter, mask_rcnn, label_names=class_names, show_progress=True)
    result = evaluator()

    for k in result:
        if isinstance(result[k], (np.float32, np.float64)):
            result[k] = float(result[k])
        else:
            assert isinstance(result[k], float), \
                'Unsupported type: {}, key: {}'.format(type(result[k]), k)

    yaml_file = pretrained_model + '.eval_result.yaml'
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print('Saved evaluation: %s' % yaml_file)
    print('# ' + '-' * 77)
    pprint.pprint(result)
    print('# ' + '-' * 77)


if __name__ == '__main__':
    main()
