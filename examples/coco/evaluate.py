#!/usr/bin/env python

import os.path as osp
import sys

import chainer_mask_rcnn as cmr

here = osp.dirname(osp.abspath(__file__))  # NOQA
sys.path.insert(0, osp.join(here, '..'))  # NOQA

import evaluate_common


def main():
    test_data = cmr.datasets.COCOInstanceSegmentationDataset(
        'minival',
        use_crowd=True,
        return_crowd=True,
        return_area=True,
    )

    evaluate_common.evaluate(
        test_data=test_data,
        evaluator_type='coco',
        indices_vis=[10, 22, 61, 104, 107, 116, 127, 149, 185],
    )


if __name__ == '__main__':
    main()
