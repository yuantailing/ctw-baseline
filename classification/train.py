# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import settings
import subprocess
import sys


env = {
    'PYTHONPATH': '.:tf_hardcode:slim',
    'CUDA_VISIBLE_DEVICES': '0',
}

cfg_common = {
    'dataset_dir': settings.TRAINVAL_PICKLE,
    'dataset_split_name': 'train',
    'num_clones': '1',
    'learning_rate_decay_type': 'exponential',
    'learning_rate': '0.01',
    'learning_rate_decay_factor': '0.1',
    'decay_steps': '40000',
    'max_number_of_steps': '100000',
    'per_process_gpu_memory_fraction': '1.0',
    'save_summaries_secs': '120',
    'save_interval_secs': '1200',
}

cfgs = [
    {
        'model_name': 'alexnet_v2',
        'train_image_size': '224',
        'batch_size': '64',
    },
    {
        'model_name': 'inception_v4',
        'train_image_size': '235',
        'batch_size': '64',
    },
    {
        'model_name': 'overfeat',
        'train_image_size': '231',
        'batch_size': '64',
    },
    {
        'model_name': 'resnet_v2_50',
        'train_image_size': '224',
        'batch_size': '64',
    },
    {
        'model_name': 'resnet_v2_152',
        'train_image_size': '224',
        'batch_size': '64',
    },
    {
        'model_name': 'vgg_16',
        'train_image_size': '224',
        'batch_size': '64',
    },
]

for cfg in cfgs:
    cfg['train_dir'] = os.path.join(settings.PRODUCTS_ROOT, 'train_logs_{}'.format(cfg['model_name']))


def main(model_name):
    cfg = copy.deepcopy(cfg_common)
    cfg_model = list(filter(lambda o: o['model_name'] == model_name, cfgs))[0]
    cfg.update(cfg_model)

    args = ['python3', 'slim/train_image_classifier.py']
    for k, v, in cfg.items():
        args.append('--{}={}'.format(k, v))

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


if __name__ == '__main__':
    main(sys.argv[1])
