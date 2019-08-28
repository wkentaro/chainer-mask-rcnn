#!/usr/bin/env python

import datetime
import json
import os
import os.path as osp
import subprocess
import yaml

import dateutil.parser
import pandas
import tabulate


def seconds_to_string(seconds):
    seconds = int(round(seconds))
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    value = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return value


def summarize_log(logs_dir, name, keys, target_key, objective):
    try:
        params = yaml.load(open(osp.join(logs_dir, name, 'params.yaml')))
    except Exception:
        return None, None, osp.join(logs_dir, name)

    log_file = osp.join(logs_dir, name, 'log')
    try:
        df = pandas.DataFrame(json.load(open(log_file)))
    except Exception:
        df = None

    try:
        if objective == 'min':
            idx = df[target_key].idxmin()
        else:
            idx = df[target_key].idxmax()
    except Exception:
        idx = None

    eval_result = None
    eval_result_file = osp.join(
        logs_dir, name, 'snapshot_model.npz.eval_result.yaml')
    if osp.exists(eval_result_file):
        eval_result = yaml.load(open(eval_result_file))

    dfi = df.ix[idx] if idx else None
    row = []
    is_active = True
    for key in keys:
        if key == 'name':
            row.append(name)
        elif key == 'elapsed_time':
            if dfi is None:
                value = '<none>'
            else:
                value = seconds_to_string(df[key].max())
            row.append(value)
        elif key in ['epoch', 'iteration']:
            if dfi is None:
                value = '<none>'
            else:
                value = '%d' % dfi[key]
            if df is None:
                row.append('<none>')
            else:
                max_value = df[key].max()
                row.append('%s /%d' % (value, max_value))
        elif key == 'git_branch':
            value = params.get('git', '<none>')
            cmd = 'git log {:s} -1 --format="%d"'.format(value)
            try:
                value = subprocess.check_output(
                    cmd, shell=True, stderr=subprocess.PIPE).strip()
                value = value.lstrip('(').rstrip(')')
                value = value.split(',')
                if 'HEAD' in value:
                    value.remove('HEAD')
                value = value[0]
            except subprocess.CalledProcessError:
                value = ''
            row.append(value or '<none>')
        elif key.endswith('/loss'):
            if dfi is None:
                value = '<none>'
            else:
                value = '%.3f' % dfi[key]
            if df is None:
                row.append('<none>')
            else:
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
                value = params.get('timestamp', '<none>')
                if value != '<none>':
                    value = dateutil.parser.parse(value)
                    value += datetime.timedelta(seconds=elapsed_time)
                    now = datetime.datetime.now()
                    value = now - value
                    if value > datetime.timedelta(minutes=10):
                        is_active = False
                    value -= datetime.timedelta(
                        microseconds=value.microseconds)
                    value = max(datetime.timedelta(seconds=0), value)
                    value = '- %s' % seconds_to_string(value.total_seconds())
            row.append(value)
        elif key == 'eval_result':
            value = None
            if eval_result is not None:
                value = '%.3f' % eval_result['validation/main/map']
            row.append(value)
        elif key in params:
            row.append(params[key])
        elif dfi is not None and key in dfi:
            row.append(dfi[key])
        else:
            row.append('<none>')
    return row, is_active, None


def _summarize_log(args):
    return summarize_log(*args)


def summarize_logs(logs_dir, keys, target_key, objective):
    assert objective in ['min', 'max']
    assert target_key in keys

    args_list = []
    for name in os.listdir(logs_dir):
        args_list.append((
            logs_dir,
            name,
            keys,
            target_key,
            objective,
        ))

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(_summarize_log, args_list)

    rows = []
    ignored = []
    for row, is_active, log_dir_ignored in results:
        if log_dir_ignored:
            ignored.append(log_dir_ignored)
            continue
        rows.append(row)

    print('logs_dir: {}\n'.format(osp.abspath(logs_dir)))

    rows = sorted(rows, key=lambda x: x[0], reverse=True)
    print(
        tabulate.tabulate(
            rows,
            headers=keys,
            floatfmt='.3f',
            tablefmt='simple',
            numalign='center',
            stralign='center',
            showindex=True,
            disable_numparse=True,
        ),
    )

    if not ignored:
        return

    print('Ignored logs:')
    for log_dir in ignored:
        print('  - %s' % log_dir)


if __name__ == '__main__':
    keys = [
        'name',
        # 'timestamp',
        'elapsed_time',
        'last_time',
        # 'dataset',
        'git_hash',
        # 'git_branch',
        'hostname',
        'model',
        # 'roi_size',
        'initializer',
        'lr',
        # 'pooling_func',
        'epoch', 'iteration',
        # 'main/loss',
        'eval_result',
        'validation/main/map',
    ]
    objective = 'max'
    summarize_logs('logs', keys, target_key=keys[-1], objective=objective)
