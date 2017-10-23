# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import settings
import subprocess

from multiprocessing import cpu_count


env = {
    'CUDA_VISIBLE_DEVICES': '0',
}


def compile_darknet():
    args = ['make', '-j{}'.format(cpu_count())]

    print(*args)
    p = subprocess.Popen(args, cwd=settings.DARKNET_ROOT, shell=False)
    p.wait()
    assert 0 == p.returncode
    

def train_yolo():
    exefile = os.path.join(settings.DARKNET_ROOT, 'darknet')
    args = [exefile, 'detector', 'train', settings.DARKNET_DATA, settings.DARKNET_CFG, settings.DARKNET_PRETRAIN]

    new_env = os.environ.copy()
    if 'CUDA_VISIBLE_DEVICES' in new_env:
        env['CUDA_VISIBLE_DEVICES'] = new_env['CUDA_VISIBLE_DEVICES']
    new_env.update(env)

    for k, v in env.items():
        print('{}={}'.format(k, v), end=' ')
    print(*args)
    p = subprocess.Popen(args, env=new_env, shell=False)
    p.wait()
    assert 0 == p.returncode


def main():
    compile_darknet()
    train_yolo()


if __name__ == '__main__':
    main()
