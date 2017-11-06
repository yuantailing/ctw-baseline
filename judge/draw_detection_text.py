# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import settings
import subprocess

from multiprocessing import cpu_count
from pythonapi import anno_tools, common_tools


def compile():
    pro = os.path.join('printtext-src', 'printtext.pro')
    cwd = os.path.dirname(settings.PRINTTEXT_EXEC)
    makefile = os.path.join(cwd, 'Makefile')

    if not os.path.isdir(cwd):
        os.makedirs(cwd)

    args = ['qmake', os.path.abspath(pro), 'CONFIG+=release']
    print(*args)
    p = subprocess.Popen(args, cwd=cwd)
    p.wait()
    assert 0 == p.returncode

    args = ['make', '-j{}'.format(cpu_count())]
    print(*args)
    p = subprocess.Popen(args, cwd=cwd)
    p.wait()
    assert 0 == p.returncode


def print_text(in_file_name, out_file_name, color, obj):
    args = [settings.PRINTTEXT_EXEC, in_file_name, out_file_name, color]
    print(*args)
    p = subprocess.Popen(args, stdin=subprocess.PIPE)
    p.communicate('{}\n'.format(common_tools.to_jsonl(obj)))
    p.wait()
    assert 0 == p.returncode


def main():
    if not os.path.isfile(settings.PRINTTEXT_EXEC):
        compile()

    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    with open(settings.TEST_DETECTION_GT) as f:
        gts = f.read().splitlines()
    with open('../detection/products/detections.jsonl') as f:
        dts = f.read().splitlines()
    assert len(gts) == len(dts)

    def gt2array(gt, draw_ignore=False):
        a = []
        for char in anno_tools.each_char(gt):
            if char['is_chinese']:
                a.append({'bbox': char['adjusted_bbox'], 'text': char['text']})
            elif draw_ignore:
                a.append({'bbox': char['adjusted_bbox'], 'text': char['text']})
        if draw_ignore:
            for char in gt['ignore']:
                a.append({'bbox': char['bbox'], 'text': ''})
        return a

    def dt2array(dt, thresh=.5):
        return list(filter(lambda o: o['score'] > thresh, dt['detections']))

    if not os.path.isdir(settings.PRINTTEXT_DRAWING_DIR):
        os.makedirs(settings.PRINTTEXT_DRAWING_DIR)
    tasks = []
    for i in range(0, len(gts), 100):
        gt = json.loads(gts[i])
        dt = json.loads(dts[i])
        image_id = gt['image_id']
        file_name = os.path.join(settings.TEST_IMAGE_DIR, gt['file_name'])
        tasks.append((
            file_name,
            os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_gt.png'.format(image_id)),
            '#f00', gt2array(gt),
        ))
        tasks.append((
            file_name,
            os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_dt.png'.format(image_id)),
            '#0f0', dt2array(dt),
        ))
    common_tools.multithreaded(print_text, tasks, num_thread=cpu_count())


if __name__ == '__main__':
    main()
