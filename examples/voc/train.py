#!/usr/bin/env python

import os.path as osp
import sys

import chainer_mask_rcnn as cmr

here = osp.dirname(osp.abspath(__file__))  # NOQA
sys.path.insert(0, osp.join(here, '..'))  # NOQA

import train_common


def main():
    args = train_common.parse_args()

    args.logs_dir = osp.join(here, 'logs')

    # Dataset.
    args.dataset = 'voc'
    train_data = cmr.datasets.SBDInstanceSegmentationDataset('train')
    test_data = cmr.datasets.SBDInstanceSegmentationDataset('val')
    args.class_names = tuple(train_data.class_names.tolist())

    # Model.
    args.min_size = 600
    args.max_size = 1000
    args.anchor_scales = (4, 8, 16, 32)

    train_common.train(
        args=args,
        train_data=train_data,
        test_data=test_data,
        evaluator_type='voc',
    )


if __name__ == '__main__':
    main()
