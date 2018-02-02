#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint

import chainer
from chainercv.datasets import voc_bbox_label_names
import numpy as np
import yaml

import chainer_mask_rcnn as mrcnn


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('log_dir', help='Log dir.')
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
args = parser.parse_args()

gpu = args.gpu
log_dir = args.log_dir
params = yaml.load(open(osp.join(log_dir, 'params.yaml')))

if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
chainer.global_config.train = False
chainer.global_config.enable_backprop = False

pretrained_model = osp.join(log_dir, 'snapshot_model.npz')

model = params['model']
if params['pooling_func'] == 'align':
    pooling_func = mrcnn.functions.roi_align_2d
elif params['pooling_func'] == 'pooling':
    pooling_func = chainer.functions.roi_pooling_2d
elif params['pooling_func'] == 'resize':
    pooling_func = mrcnn.functions.crop_and_resize
else:
    raise ValueError
pprint.pprint(params)


n_layers = int(model.lstrip('resnet'))
proposal_creator_params = dict(
    n_train_pre_nms=12000,
    n_train_post_nms=2000,
    n_test_pre_nms=6000,
    n_test_post_nms=1000,
    min_size=0,
)
anchor_scales = [8, 16, 32]
mask_rcnn = mrcnn.models.MaskRCNNResNet(
    n_layers=n_layers,
    n_fg_class=len(voc_bbox_label_names),
    pretrained_model=pretrained_model,
    pooling_func=pooling_func,
    min_size=600,
    max_size=1000,
    proposal_creator_params=proposal_creator_params,
    anchor_scales=anchor_scales,
)
if gpu >= 0:
    mask_rcnn.to_gpu()


def transform_test_data(in_data):
    img = in_data[0]
    img = img.transpose(2, 0, 1)
    out_data = list(in_data)
    out_data[0] = img
    return tuple(out_data)


# test_data = mrcnn.datasets.MaskRcnnDataset(
#     mrcnn.datasets.VOC2012InstanceSeg('val'))
test_data = mrcnn.datasets.MaskRcnnDataset(
    mrcnn.datasets.SBDInstanceSeg('val'))
test_data = chainer.datasets.TransformDataset(test_data, transform_test_data)
test_iter = chainer.iterators.SerialIterator(
    test_data, batch_size=1, repeat=False, shuffle=False)

test_vis_data = mrcnn.datasets.IndexingDataset(
    test_data, indices=[196, 204, 216, 257, 326, 473, 566, 649, 1063])
test_vis_iter = chainer.iterators.SerialIterator(
    test_vis_data, batch_size=1, repeat=False, shuffle=False)


class DummyTrainer(object):

    class DummyUpdater(object):

        iteration = 'best'

    updater = DummyUpdater()
    out = log_dir


print('visualization:', osp.join(log_dir, 'iteration=best.jpg'))
visualizer = mrcnn.extensions.InstanceSegmentationVisReport(
    test_vis_iter, mask_rcnn,
    label_names=voc_bbox_label_names,
    file_name='iteration=%s.jpg',
    copy_latest=False,
)
visualizer(trainer=DummyTrainer())

print('evaluation:')
mask_rcnn.use_preset('evaluate')
evaluator = mrcnn.extensions.InstanceSegmentationVOCEvaluator(
    test_iter, mask_rcnn, use_07_metric=True,
    label_names=voc_bbox_label_names)
result = evaluator()

for k in result:
    if isinstance(result[k], np.float64):
        result[k] = float(result[k])

yaml_file = osp.join(log_dir, 'result_evaluate.yaml')
with open(yaml_file, 'w') as f:
    yaml.safe_dump(result, f, default_flow_style=False)

print('saved result: %s' % yaml_file)
pprint.pprint(result)
