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
    row = [name]
    row += [dfi[c] for c in columns[1:]]
    rows.append(row)
rows = sorted(rows, key=lambda x: x[4])
print(tabulate.tabulate(rows, columns, tablefmt='grid'))
