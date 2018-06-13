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
    parser.add_argument('log_dir', help='Log dir.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    args = parser.parse_args()

    test_data = cmr.datasets.SBDInstanceSegmentationDataset('val')

    evaluate(
        gpu=args.gpu,
        log_dir=args.log_dir,
        test_data=test_data,
        evaluator_type='voc',
        indices_vis=[196, 204, 216, 257, 326, 473, 566, 649, 1063],
    )


if __name__ == '__main__':
    main()
