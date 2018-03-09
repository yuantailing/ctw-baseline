# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import operator
import os
import settings
import tarfile


def reset(tarinfo):
    tarinfo.name = os.path.basename(tarinfo.name)
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = 'root'
    tarinfo.mtime = 0
    tarinfo.mode = 0o644
    return tarinfo


def main():
    with tarfile.open(os.path.join(settings.PRODUCTS_ROOT, 'ctw-annotations.tar.gz'), 'w|gz') as tar:
        tar.add(settings.DATA_LIST, filter=reset)
        tar.add(settings.TRAIN, filter=reset)
        tar.add(settings.VAL, filter=reset)
        tar.add(settings.TEST_CLASSIFICATION, filter=reset)
    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    all = (('trainval', data_list['train'] + data_list['val'], settings.TRAINVAL_IMAGE_DIR),
           ('test', data_list['test_cls'] + data_list['test_det'], settings.TEST_IMAGE_DIR))
    for prefix, meta, src_dir in all:
        meta.sort(key=operator.itemgetter('image_id'))
        delta = 1000
        n = len(range(0, len(meta), delta))
        for i in range(0, len(meta), delta):
            submeta = meta[i:i + delta]
            with tarfile.open(os.path.join(settings.PRODUCTS_ROOT, 'ctw-{}-{:02d}-of-{:02d}.tar'.format(prefix, i // delta + 1, n)), 'w') as tar:
                for o in submeta:
                    tar.add(os.readlink(os.path.join(src_dir, o['file_name'])), filter=reset)


if __name__ == '__main__':
    main()
