#!/usr/bin/env python

import math
import os
import os.path as osp

import pandas as pd
import tabulate


def summarize_logs(logs_dir, keys, target_key, objective):
    assert objective in ['min', 'max']
    assert target_key in keys

    rows = []
    for name in os.listdir(logs_dir):
        log_file = osp.join(logs_dir, name, 'log')
        name_n_rows = int(math.ceil(len(name) / 79.))
        name = '\n'.join(name[i * 79:(i + 1) * 79] for i in range(name_n_rows))
        try:
            df = pd.read_json(log_file)
            if objective == 'min':
                idx = df[target_key].idxmin()
            else:
                idx = df[target_key].idxmax()
        except Exception:
            continue
        dfi = df.ix[idx]
        # if dfi['validation/main/loss'] == 0:
        #     continue
        row = []
        for key in keys:
            if key == 'name':
                row.append(name)
            elif key in ['epoch', 'iteration']:
                max_value = df[key].max()
                row.append('%d/%d' % (dfi[key], max_value))
            elif key.endswith('/loss'):
                min_value = df[key].min()
                max_value = df[key].max()
                row.append('%.3f<%.3f<%.3f' %
                           (min_value, dfi[key], max_value))
            else:
                row.append(dfi[key])
        rows.append(row)
    rows = sorted(rows, key=lambda x: x[4], reverse=objective == 'min')
    print(tabulate.tabulate(rows, headers=keys,
                            floatfmt='.2f', tablefmt='grid'))


if __name__ == '__main__':
    keys = ['name', 'epoch', 'iteration', 'main/loss', 'validation/main/loss']
    objective = 'min'
    summarize_logs('logs', keys, target_key=keys[-1], objective=objective)
