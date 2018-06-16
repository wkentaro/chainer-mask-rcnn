from __future__ import print_function

import argparse
import os.path as osp
import pprint

import chainer
import numpy as np
import yaml

import chainer_mask_rcnn as cmr


def evaluate(test_data, evaluator_type, indices_vis=None):
    assert evaluator_type in ['voc', 'coco'], \
        'Unsupported evaluator_type: {}'.format(evaluator_type)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    # param
    with open(osp.join(args.log_dir, 'params.yaml')) as f:
        params = yaml.load(f)
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    # dataset
    class_names = test_data.class_names

    # model

    if params['pooling_func'] == 'align':
        pooling_func = cmr.functions.roi_align_2d
    elif params['pooling_func'] == 'pooling':
        pooling_func = cmr.functions.roi_pooling_2d
    elif params['pooling_func'] == 'resize':
        pooling_func = cmr.functions.crop_and_resize
    else:
        raise ValueError

    pretrained_model = osp.join(args.log_dir, 'snapshot_model.npz')
    print('Using pretrained_model:', pretrained_model)

    model = params['model']
    mask_rcnn = cmr.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        anchor_scales=params['anchor_scales'],
        mean=params.get('mean', (123.152, 115.903, 103.063)),
        min_size=params['min_size'],
        max_size=params['max_size'],
        roi_size=params['roi_size'],
    )
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()

    test_data = chainer.datasets.TransformDataset(
        test_data,
        cmr.datasets.MaskRCNNTransform(mask_rcnn, train=False),
    )

    # visualization
    # -------------------------------------------------------------------------

    test_vis_data = cmr.datasets.IndexingDataset(
        test_data,
        indices=indices_vis,
    )
    test_vis_iter = chainer.iterators.SerialIterator(
        test_vis_data,
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    class DummyTrainer(object):

        class DummyUpdater(object):

            iteration = 'best'

        updater = DummyUpdater()
        out = args.log_dir

    print('Visualizing...')
    visualizer = cmr.extensions.InstanceSegmentationVisReport(
        iterator=test_vis_iter,
        target=mask_rcnn,
        label_names=class_names,
        file_name='iteration=%s.jpg',
        copy_latest=False,
    )
    visualizer(trainer=DummyTrainer())
    print('Saved visualization:', osp.join(args.log_dir, 'iteration=best.jpg'))

    # evaluation
    # -------------------------------------------------------------------------

    test_iter = chainer.iterators.SerialIterator(
        test_data,
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    print('Evaluating...')
    if evaluator_type == 'voc':
        evaluator = cmr.extensions.InstanceSegmentationVOCEvaluator(
            test_iter,
            mask_rcnn,
            use_07_metric=True,
            label_names=class_names,
            show_progress=True,
        )
    elif evaluator_type == 'coco':
        evaluator = cmr.extensions.InstanceSegmentationCOCOEvaluator(
            test_iter,
            mask_rcnn,
            label_names=class_names,
            show_progress=True,
        )
    else:
        raise ValueError('Unsupported evaluator type: %s' % evaluator_type)
    result = evaluator()

    for k in result:
        if isinstance(result[k], np.floating):
            result[k] = float(result[k])

    yaml_file = pretrained_model + '.eval_result.yaml'
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print('Saved evaluation:', yaml_file)
    pprint.pprint(result)
