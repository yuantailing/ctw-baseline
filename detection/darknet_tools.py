# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import imp
import math
import operator
import os
import re
import settings
import subprocess

from multiprocessing import cpu_count


def compile_darknet():
    args = ['make', '-j{}'.format(cpu_count())]

    print(*args)
    p = subprocess.Popen(args, cwd=settings.DARKNET_ROOT, shell=False)
    p.wait()
    assert 0 == p.returncode


def import_darknet():
    return imp.load_source('darknet', os.path.join(settings.DARKNET_ROOT, 'python', 'darknet.py'))


def last_backup(backup_root):
    if not os.path.isdir(backup_root):
        return None
    basename = os.path.splitext(os.path.basename(settings.DARKNET_CFG))[0]
    final_weights = os.path.join(backup_root, '{}_final.weights'.format(basename))
    if os.path.isfile(final_weights):
        return final_weights
    r = re.compile(r'^{}_(\d+)\.weights$'.format(re.escape(basename)))
    all = [(None, -1)]
    for filename in os.listdir(backup_root):
        filepath = os.path.join(backup_root, filename)
        if os.path.isfile(filepath):
            m = r.match(filename)
            if m:
                i = int(m.group(1))
                all.append((filepath, i))
    return max(all, key=operator.itemgetter(1))[0]


def get_crop_bboxes(imshape, cropshape, cropoverlap):
    crop_num_y = int(math.ceil((imshape[0] - cropshape[0]) / (cropshape[0] - cropoverlap[0]) + 1))
    crop_num_x = int(math.ceil((imshape[1] - cropshape[1]) / (cropshape[1] - cropoverlap[1]) + 1))
    for i in range(crop_num_y):
        for j in range(crop_num_x):
            ylo = int(round(i * (imshape[0] - cropshape[0]) / (crop_num_y - 1)))
            xlo = int(round(j * (imshape[1] - cropshape[1]) / (crop_num_x - 1)))
            yield {'name': '{}_{}'.format(i, j), 'xlo': xlo, 'ylo': ylo}


def append_before_ext(filepath, s):
    base, ext = os.path.splitext(filepath)
    return '{}{}{}'.format(base, s, ext)
