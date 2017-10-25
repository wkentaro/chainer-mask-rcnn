#!/usr/bin/env python

import os
import os.path as osp

import pandas as pd
import tabulate


def split_name(name):
    splits = name.split('.')
    splits_new = []
    for kv in splits:
        if '=' in kv:
            splits_new.append(kv)
        else:
            splits_new[-1] += kv
    return splits_new


def summarize_logs(logs_dir, keys, target_key, objective):
    assert objective in ['min', 'max']
    assert target_key in keys

    rows = []
    for name in os.listdir(logs_dir):
        log_file = osp.join(logs_dir, name, 'log')
        try:
            df = pd.read_json(log_file)
            if objective == 'min':
                idx = df[target_key].idxmin()
            else:
                idx = df[target_key].idxmax()
        except Exception:
            idx = None
        dfi = df.ix[idx] if idx else None
        row = []
        for key in keys:
            if key == 'name':
                row.append(name)
            elif key in ['epoch', 'iteration']:
                if dfi is None:
                    row.append('<none>')
                else:
                    max_value = df[key].max()
                    row.append('%d /%d' % (dfi[key], max_value))
            elif key.endswith('/loss'):
                if dfi is None:
                    row.append('<none>')
                else:
                    min_value = df[key].min()
                    max_value = df[key].max()
                    row.append('%.3f< %.3f <%.3f' %
                               (min_value, dfi[key], max_value))
            elif key.endswith('/map'):
                if dfi is None:
                    row.append('<none>')
                else:
                    if objective == 'max':
                        min_value = df[key].min()
                        row.append('%.3f< %.3f' % (min_value, dfi[key]))
                    else:
                        max_value = df[key].max()
                        row.append('%.3f <%.3f' % (dfi[key], max_value))
            elif dfi is not None and key in dfi:
                row.append(dfi[key])
            else:
                value = '<none>'
                for kv in split_name(name):
                    k, v = kv.split('=')
                    if k == key:
                        value = v
                row.append(value)
        rows.append(row)
    rows = sorted(rows, key=lambda x: x[0], reverse=objective == 'min')
    print(tabulate.tabulate(rows, headers=keys,
                            floatfmt='.3f', tablefmt='grid',
                            numalign='center', stralign='center',
                            showindex=True))


if __name__ == '__main__':
    keys = [
        'timestamp', 'hostname', 'dataset', 'model', 'pretrained_model',
        'update_policy', 'pooling_func',
        'epoch', 'iteration',
        'main/loss', 'validation/main/map',
    ]
    objective = 'max'
    summarize_logs('logs', keys, target_key=keys[-1], objective=objective)
