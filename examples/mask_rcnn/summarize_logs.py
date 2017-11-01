#!/usr/bin/env python

import argparse
import datetime
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


def summarize_logs(logs_dir, keys, target_key, objective, show_active):
    assert objective in ['min', 'max']
    assert target_key in keys

    rows = []
    ignored = []
    for name in os.listdir(logs_dir):
        log_file = osp.join(logs_dir, name, 'log')
        try:
            df = pd.read_json(log_file)
        except Exception:
            ignored.append(osp.join(logs_dir, name))
            continue

        try:
            if objective == 'min':
                idx = df[target_key].idxmin()
            else:
                idx = df[target_key].idxmax()
        except Exception:
            idx = None

        dfi = df.ix[idx] if idx else None
        row = []
        is_active = True
        for key in keys:
            if key == 'name':
                row.append(name)
            elif key in ['epoch', 'iteration']:
                if dfi is None:
                    value = '<none>'
                else:
                    value = '%d' % dfi[key]
                max_value = df[key].max()
                row.append('%s /%d' % (value, max_value))
            elif key.endswith('/loss'):
                if dfi is None:
                    value = '<none>'
                else:
                    value = '%.3f' % dfi[key]
                min_value = df[key].min()
                max_value = df[key].max()
                row.append('%.3f< %s <%.3f' % (min_value, value, max_value))
            elif key.endswith('/map'):
                min_value = max_value = '<none>'
                if dfi is None:
                    value = '<none>'
                else:
                    value = '%.3f' % dfi[key]
                if objective == 'max':
                    if df is not None and key in df:
                        min_value = '%.3f' % df[key].min()
                    row.append('%s< %s' % (min_value, value))
                else:
                    if df is not None and key in df:
                        max_value = '%.3f' % df[key].max()
                    row.append('%s <%s' % (value, max_value))
            elif key == 'last_time':
                if df is None:
                    value = '<none>'
                else:
                    elapsed_time = df['elapsed_time'].max()

                    value = '<none>'
                    key = 'timestamp'
                    for kv in split_name(name):
                        k, v = kv.split('=')
                        if k == key:
                            value = v
                    if value is not '<none>':
                        value = datetime.datetime.strptime(
                            value, '%Y%m%d_%H%M%S')
                        value += datetime.timedelta(seconds=elapsed_time)
                        now = datetime.datetime.now()
                        value = now - value
                        if value > datetime.timedelta(minutes=10):
                            is_active = False
                        value -= datetime.timedelta(
                            microseconds=value.microseconds)
                        value = '- %s' % value.__str__()
                row.append(value)
            elif dfi is not None and key in dfi:
                row.append(dfi[key])
            else:
                value = '<none>'
                for kv in split_name(name):
                    k, v = kv.split('=')
                    if k == key:
                        value = v
                row.append(value)
        if show_active:
            if is_active:
                rows.append(row)
        else:
            rows.append(row)
    rows = sorted(rows, key=lambda x: x[0], reverse=True)
    print(tabulate.tabulate(rows, headers=keys,
                            floatfmt='.3f', tablefmt='grid',
                            numalign='center', stralign='center',
                            showindex=True))

    if not ignored:
        return

    print('Ignored logs:')
    for log_dir in ignored:
        print('  - %s' % log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--active', action='store_true')
    args = parser.parse_args()

    keys = [
        'timestamp', 'last_time',
        'git', 'hostname', 'model', 'pooling_func',
        'epoch', 'iteration',
        'main/loss', 'validation/main/map',
    ]
    objective = 'max'
    summarize_logs('logs', keys, target_key=keys[-1],
                   objective=objective, show_active=args.active)
