# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import settings
import shutil
import threading

from pythonapi import common_tools


def main():
    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    m = dict()
    for root in settings.IMAGE_SOURCES_ROOTS:
        for p, r, fn in common_tools.each_file_tuple(root):
            m[fn] = p

    def cp(file_name):
        src = m[file_name]
        dst = os.path.join(copy_dest, file_name)
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    copy_dest = settings.TRAINVAL_IMAGE_DIR
    if os.path.isdir(copy_dest):
        shutil.rmtree(copy_dest)
    os.makedirs(copy_dest)
    file_names = list({meta['file_name'] for meta in data_list['train'] + data_list['val']})
    common_tools.multithreaded(cp, file_names, num_thread=8)

    copy_dest = settings.TEST_IMAGE_DIR
    if os.path.isdir(copy_dest):
        shutil.rmtree(copy_dest)
    os.makedirs(copy_dest)
    file_names = list({meta['file_name'] for meta in data_list['test_cls'] + data_list['test_det']})
    common_tools.multithreaded(cp, file_names, num_thread=8)

if __name__ == '__main__':
    main()
