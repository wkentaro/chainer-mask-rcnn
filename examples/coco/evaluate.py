#!/usr/bin/env python

import argparse

import os.path as osp
import sys

import chainer_mask_rcnn as cmr

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '..'))

from evaluate import evaluate  # NOQA


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    test_data = cmr.datasets.COCOInstanceSegmentationDataset(
        'minival',
        use_crowd=True,
        return_crowd=True,
        return_area=True,
    )

    evaluate(
        gpu=args.gpu,
        log_dir=args.log_dir,
        test_data=test_data,
        evaluator_type='coco',
        indices_vis=[10, 22, 61, 104, 107, 116, 127, 149, 185],
    )


if __name__ == '__main__':
    main()
