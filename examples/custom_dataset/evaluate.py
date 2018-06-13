#!/usr/bin/env python

import os.path as osp
import sys

from train import VOCLikeDataset

here = osp.dirname(osp.abspath(__file__))  # NOQA
sys.path.insert(0, osp.join(here, '..'))  # NOQA

import evaluate_common


def main():
    root_dir = osp.join(
        here, 'src/labelme/examples/instance_segmentation/data_dataset_voc',
    )
    test_data = VOCLikeDataset(root_dir=root_dir)

    evaluate_common.evaluate(
        test_data=test_data,
        evaluator_type='voc',
        indices_vis=[0, 1, 2],
    )


if __name__ == '__main__':
    main()
