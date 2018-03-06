# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import settings
import subprocess


def main():
    assert os.path.isfile(settings.PRETRAIN_MODEL), 'please download pretrain model {}'.format(settings.PRETRAIN_MODEL)
    assert os.path.isfile(os.path.join(settings.CAFFE_ROOT, 'build/tools/caffe')), 'please build caffe'

    exe = 'python2'
    script = os.path.join('ssd_hardcode', 'ssd_pascal_512.py')
    args = [exe, script]

    env = os.environ.copy()
    pythonpath = os.path.join(settings.CAFFE_ROOT, 'python')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = '{}:{}:{}'.format(env['PYTHONPATH'], '.', pythonpath)
    else:
        env['PYTHONPATH'] = '{}:{}'.format('.', pythonpath)

    print(*args)
    p = subprocess.Popen(args, env=env)
    p.wait()
    assert 0 == p.returncode


if __name__ == '__main__':
    main()
