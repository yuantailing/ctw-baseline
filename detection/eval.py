# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import darknet_tools
import os
import settings
import subprocess


env = {
    'CUDA_VISIBLE_DEVICES': '0',
}


def train_yolo():
    exefile = os.path.join(settings.DARKNET_ROOT, 'darknet')
    last_backup = darknet_tools.last_backup(settings.DARKNET_BACKUP_DIR)
    assert last_backup is not None
    args = [exefile, 'detector', 'valid', settings.DARKNET_DATA, settings.DARKNET_TEST_CFG,
            last_backup]

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
    darknet_tools.compile_darknet()
    train_yolo()


if __name__ == '__main__':
    main()
