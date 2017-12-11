# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import settings
import subprocess


def main():
    if not os.path.isdir(settings.CODALAB_TEST_REF):
        os.makedirs(settings.CODALAB_TEST_REF)
    meta_path = os.path.join(settings.CODALAB_TEST_REF, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'task': 'classification',
            'split': 'test_cls',
            'aes_key': None,
        }, f)

    if not os.path.isdir(settings.CODALAB_TEST_OUTPUT):
        os.makedirs(settings.CODALAB_TEST_OUTPUT)
    args = ['python2', 'evaluate.py', '../classification/products/predictions_inception_v4.jsonl', settings.CODALAB_TEST_REF, settings.CODALAB_TEST_OUTPUT]
    print(*args)
    p = subprocess.Popen(args)
    return p.wait()


if __name__ == '__main__':
    exit(main())
