#!/usr/bin/env python

import os.path as osp
import sys

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '../voc'))
from summarize_logs import summarize_logs


if __name__ == '__main__':
    keys = [
        'name',
        'elapsed_time',
        # 'timestamp',
        'last_time',
        'dataset',
        'git_hash',
        # 'git_branch',
        'hostname',
        'model',
        'lr', 'pooling_func',
        'epoch', 'iteration',
        # 'main/loss',
        'validation/main/map',
    ]
    objective = 'max'
    summarize_logs('logs', keys, target_key=keys[-1], objective=objective)
