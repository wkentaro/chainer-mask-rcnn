#!/usr/bin/env python

import pandas as pd
import tabulate

import os
import os.path as osp


columns = ['name', 'epoch', 'iteration', 'main/loss', 'validation/main/map']
rows = []
for name in os.listdir('logs'):
    log_file = osp.join('logs', name, 'log')
    name = name[:len(name) // 2] + '\n' + name[len(name) // 2:]
    try:
        df = pd.read_json(log_file)
        idx = df['validation/main/map'].idxmax()
    except Exception:
        continue
    dfi = df.ix[idx]
    if dfi['validation/main/map'] == 0:
        continue
    row = []
    for col in columns:
        if col == 'name':
            row.append(name)
        elif col in ['epoch', 'iteration']:
            max_value = df[col].max()
            row.append('%d / %d' % (dfi[col], max_value))
        elif col == 'main/loss':
            min_value = df[col].min()
            max_value = df[col].max()
            row.append('%f < %f < %f' % (min_value, dfi[col], max_value))
        else:
            row.append(dfi[col])
    rows.append(row)
rows = sorted(rows, key=lambda x: x[4])
print(tabulate.tabulate(rows, columns, tablefmt='grid'))
