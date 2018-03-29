# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import darknet_tools
import os
import settings
import subprocess

from pythonapi import common_tools


env = {
    'CUDA_VISIBLE_DEVICES': '0',
}


def write_darknet_test_data(split_id):
    darknet_valid_list = darknet_tools.append_before_ext(settings.DARKNET_VALID_LIST, '.{}'.format(split_id))
    with open(settings.DARKNET_VALID_LIST) as f:
        ls = f.read().splitlines()
    with open(darknet_valid_list, 'w') as f:
        for line in ls[split_id * len(ls) // settings.TEST_SPLIT_NUM:(1 + split_id) * len(ls) // settings.TEST_SPLIT_NUM]:
            f.write(line)
            f.write('\n')

    darknet_data = darknet_tools.append_before_ext(settings.DARKNET_DATA, '.{}'.format(split_id))
    data = {
        'classes': settings.NUM_CHAR_CATES + 1,
        'valid': darknet_valid_list,
        'names': settings.DARKNET_NAMES,
        'results': settings.DARKNET_RESULTS_DIR,
        'eval': 'chinese',
    }
    with open(darknet_data, 'w') as f:
        for k, v in sorted(data.items()):
            f.write('{} = {}\n'.format(k, v))


def eval_yolo(split_id, tid):
    exefile = os.path.join(settings.DARKNET_ROOT, 'darknet')
    last_backup = darknet_tools.last_backup(settings.DARKNET_BACKUP_DIR)
    assert last_backup is not None
    darknet_data = darknet_tools.append_before_ext(settings.DARKNET_DATA, '.{}'.format(split_id))
    if not os.path.isdir(os.path.dirname(settings.DARKNET_RESULTS_OUT)):
        os.makedirs(os.path.dirname(settings.DARKNET_RESULTS_OUT))
    darknet_results_out = darknet_tools.append_before_ext(settings.DARKNET_RESULTS_OUT, '.{}'.format(split_id))

    args = [exefile, 'detector', 'valid', darknet_data, settings.DARKNET_TEST_CFG,
            last_backup, '-out', darknet_results_out]

    new_env = os.environ.copy()
    if 'CUDA_VISIBLE_DEVICES' in new_env:
        env['CUDA_VISIBLE_DEVICES'] = new_env['CUDA_VISIBLE_DEVICES']
    if 1 != settings.TEST_NUM_GPU:
        env['CUDA_VISIBLE_DEVICES'] = '{}'.format(tid % settings.TEST_NUM_GPU)
    new_env.update(env)

    for k, v in env.items():
        print('{}={}'.format(k, v), end=' ')
    print(*args)
    p = subprocess.Popen(args, env=new_env, shell=False)
    p.wait()
    assert 0 == p.returncode


def main():
    if not os.path.exists(settings.DARKNET_RESULTS_DIR):
        os.makedirs(settings.DARKNET_RESULTS_DIR)
    for i in range(settings.TEST_SPLIT_NUM):
        write_darknet_test_data(i)
    darknet_tools.compile_darknet()
    common_tools.multithreaded_tid(eval_yolo, range(settings.TEST_SPLIT_NUM), num_thread=6)


if __name__ == '__main__':
    main()
