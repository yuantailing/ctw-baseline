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

    def cp(desc):
        file_name = desc['file_name']
        assert file_name in m
        shutil.copy(m[file_name], os.path.join(copy_dest, file_name))

    copy_dest = settings.TRAINVAL_IMAGE_DIR
    common_tools.mkdirs(copy_dest)
    common_tools.multithreaded(cp, data_list['train'] + data_list['val'], num_thread=8)

    copy_dest = settings.TEST_IMAGE_DIR
    common_tools.mkdirs(copy_dest)
    common_tools.multithreaded(cp, data_list['test_cls'] + data_list['test_det'], num_thread=8)

if __name__ == '__main__':
    main()
