# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import settings
import six
import subprocess


env = {
    'PYTHONPATH': '.:tf_hardcode:slim',
    'CUDA_VISIBLE_DEVICES': '0',
}

cfgs = [
    {
        'model_name': 'alexnet_v2',
        'train_dir': 'products/train_logs_alexnet_v2',
        'train_image_size': '224',
        'batch_size': '64',
    }
]

cfg_common = {
    'dataset_dir': settings.TRAINVAL_PICKLE,
    'dataset_split_name': 'train',
    'preprocessing_name': 'lenet',
    'num_clones': '1',
    'learning_rate_decay_type': 'exponential',
    'learning_rate': '0.01',
    'learning_rate_decay_factor': '0.1',
    'num_epochs_per_decay': '4.0',
    'max_number_of_steps': '100000',
    'per_process_gpu_memory_fraction': '0.8',
    'save_summaries_secs': '60',
    'save_interval_secs': '1200',
}


def main(model_name):
    cfg = list(filter(lambda o: o['model_name'] == model_name, cfgs))[0]
    cfg.update(cfg_common)

    args = ['python3', 'slim/train_image_classifier.py']
    for k, v, in cfg.items():
        args.append('--{}={}'.format(k, v))

    new_env = os.environ.copy()
    new_env.update(env)

    for k, v in env.items():
        print('{}={}'.format(k, v), end=' ')
    print(*args)
    p = subprocess.Popen(args, env=new_env, shell=False)
    p.wait()


if __name__ == '__main__':
    main('alexnet_v2')
