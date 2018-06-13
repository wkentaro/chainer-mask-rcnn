#!/usr/bin/env python

import os.path as osp
import sys

import chainer_mask_rcnn as cmr

here = osp.dirname(osp.abspath(__file__))  # NOQA
sys.path.insert(0, osp.join(here, '..'))  # NOQA

import evaluate_common


def main():
    test_data = cmr.datasets.SBDInstanceSegmentationDataset('val')

    evaluate_common.evaluate(
        test_data=test_data,
        evaluator_type='voc',
        indices_vis=[196, 204, 216, 257, 326, 473, 566, 649, 1063],
    )


if __name__ == '__main__':
    main()
