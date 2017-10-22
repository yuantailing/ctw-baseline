# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import predict
import settings
import subprocess
import sys
import train


env = {
    'CUDA_VISIBLE_DEVICES': '0',
}

cfg_common = {
    'dataset_dir': settings.TEST_PICKLE,
    'dataset_split_name': 'test',
    'batch_size': 256,
}

cfgs = [{
    'model_name': cfg['model_name'],
    'eval_dir': os.path.join(settings.PRODUCTS_ROOT, 'eval_{}.pkl'.format(cfg['model_name'])),
    'checkpoint_path': cfg['train_dir'],
    'eval_image_size': cfg['train_image_size'],
} for cfg in train.cfgs]


def main(model_name):
    cfg = copy.deepcopy(cfg_common)
    cfg_model = list(filter(lambda o: o['model_name'] == model_name, cfgs))[0]
    cfg.update(cfg_model)

    args = ['python3', 'slim/eval_image_classifier.py', '--alsologtostderr']
    for k, v, in cfg.items():
        args.append('--{}={}'.format(k, v))

    new_env = os.environ.copy()
    if 'CUDA_VISIBLE_DEVICES' in new_env:
        env['CUDA_VISIBLE_DEVICES'] = new_env['CUDA_VISIBLE_DEVICES']
    new_env.update(train.env)
    new_env.update(env)

    for k, v in env.items():
        print('{}={}'.format(k, v), end=' ')
    print(*args)
    p = subprocess.Popen(args, env=new_env, shell=False)
    p.wait()
    assert 0 == p.returncode


if __name__ == '__main__':
    main(sys.argv[1])
    predict.main(sys.argv[1], 5)
