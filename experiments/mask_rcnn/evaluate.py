#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint

import chainer
from chainercv.datasets import voc_bbox_label_names
import yaml

import chainer_mask_rcnn as mrcnn

from train import InstanceSegmentationVisReport
from train import InstanceSegmentationVOCEvaluator
from train import OverfitDataset
from train import TransformDataset


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('log_dir', help='Log dir.')
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
args = parser.parse_args()

gpu = args.gpu
log_dir = args.log_dir
params = yaml.load(open(osp.join(log_dir, 'params.yaml')))

gpu = 0
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
mask_rcnn = mrcnn.models.MaskRCNNResNet(
    n_layers=n_layers,
    n_fg_class=len(voc_bbox_label_names),
    pretrained_model=pretrained_model,
    pooling_func=pooling_func)
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
test_data = TransformDataset(test_data, transform_test_data)
test_iter = chainer.iterators.SerialIterator(
    test_data, batch_size=1, repeat=False, shuffle=False)

test_vis_data = OverfitDataset(
    test_data, indices=[4, 22, 26, 34, 35, 39, 42, 44, 52])
test_vis_iter = chainer.iterators.SerialIterator(
    test_vis_data, batch_size=1, repeat=False, shuffle=False)


class DummyTrainer(object):

    class DummyUpdater(object):

        iteration = 'best'

    updater = DummyUpdater()
    out = log_dir


print('visualization:', osp.join(log_dir, 'iteration=best.jpg'))
visualizer = InstanceSegmentationVisReport(
    test_vis_iter, mask_rcnn,
    label_names=voc_bbox_label_names,
    file_name='iteration=%s.jpg'
)
visualizer(trainer=DummyTrainer())

print('evaluation:')
mask_rcnn.use_preset('evaluate')
evaluator = InstanceSegmentationVOCEvaluator(
    test_iter, mask_rcnn, use_07_metric=True,
    label_names=voc_bbox_label_names)
result = evaluator()

output = pprint.pformat(result)
with open(osp.join(log_dir, 'evaluate.py.txt'), 'w') as f:
    f.write(output + '\n')
pprint.pprint(output)
