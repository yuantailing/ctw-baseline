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
from pythonapi import anno_tools, common_tools, eval_tools


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


def print_text(in_file_name, out_file_name, obj):
    args = [settings.PRINTTEXT_EXEC, in_file_name, out_file_name]
    print(*args)
    p = subprocess.Popen(args, stdin=subprocess.PIPE)
    p.communicate('{}\n'.format(common_tools.to_jsonl(obj)).encode())
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

    gtdict = dict()
    dtdict = dict()
    for gt, dt in zip(gts, dts):
        gtobj = json.loads(gt.strip())
        dtobj = json.loads(dt.strip())
        image_id = gtobj['image_id']
        gtdict[image_id] = gtobj
        dtdict[image_id] = dtobj

    def gt2array(gt, draw_ignore):
        color = '#f00'
        color_ignore = '#ff0'
        a = list()
        for char in anno_tools.each_char(gt):
            if char['is_chinese']:
                a.append({'bbox': char['adjusted_bbox'], 'text': char['text'], 'color': color})
        if draw_ignore:
            for char in gt['ignore']:
                a.append({'bbox': char['bbox'], 'text': '', 'color': color_ignore})
        return a

    def dt2array(dt, gtobj, draw_ignore):
        iou_thresh = settings.IOU_THRESH
        charset = set()
        proposal = False

        def in_size(_):
            return True

        dt = dt['detections']
        dt.sort(key=lambda o: -o['score'])  # sort must be stable, otherwise mAP will be slightly different
        dt = [(o['bbox'], o.get('text'), o['score']) for o in dt]

        ig = [(o['bbox'], None) for o in gtobj['ignore']]
        gt = []
        for char in anno_tools.each_char(gtobj):
            if char['is_chinese']:
                charset.add(char['text'])
                gt.append((char['adjusted_bbox'], char['text'], char['properties']))

        dt_matches = [[] for i in range(len(dt))]
        dt_ig = [False] * len(dt)
        for i_dt, dtchar in enumerate(dt):
            for i_gt, gtchar in enumerate(gt):
                if proposal or dtchar[1] == gtchar[1]:
                    miou = eval_tools.iou(dtchar[0], gtchar[0])
                    if miou > iou_thresh:
                        dt_matches[i_dt].append((i_gt, miou))
            for igchar in ig:
                miou = eval_tools.a_in_b(dtchar[0], igchar[0])
                if miou > iou_thresh:
                    dt_ig[i_dt] = True
        for matches in dt_matches:
            matches.sort(key=lambda t: -t[1])  # sort must be stable, otherwise you shoule use key=lambda t: (-t[1], t[0])

        dt_matched = [0 if in_size(o[0]) and False == b else 2 for o, b in zip(dt, dt_ig)]
        gt_taken = [(0, None) if in_size(o[0]) else (2, None) for o in gt]
        for i_dt, matches in enumerate(dt_matches):
            for i_gt, _ in matches:
                if 1 != dt_matched[i_dt] and 1 != gt_taken[i_gt][0]:
                    if 0 == gt_taken[i_gt][0]:
                        dt_matched[i_dt] = 1
                        gt_taken[i_gt] = (1, i_dt)
                    else:
                        dt_matched[i_dt] = 2

        allowed = len(gt)
        dt_topnum = 0
        for ms in enumerate(dt_matched):
            if not allowed:
                break
            dt_topnum += 1
            if ms != 2:
                allowed -= 1

        a = list()
        colormap = {0: '#0ff', 1: '#0f0', 2: '#ff0'}
        for i in range(dt_topnum):
            bbox, text, score = dt[i]
            taken = dt_matched[i]
            if 2 != taken or draw_ignore:
                a.append({'bbox': bbox, 'text': text or '■', 'color': colormap[taken]})
        return a

    def remove_outside(drawlist, crop):
        def foo(o):
            x, y, w, h = o['bbox']
            cx, cy, cw, ch = crop
            if x + w < cx or x >= cx + cw or y + h < cy or y >= cy + ch:
                return False
            return True
        return list(filter(foo, drawlist))

    selected = gtdict.keys()

    if not os.path.isdir(settings.PRINTTEXT_DRAWING_DIR):
        os.makedirs(settings.PRINTTEXT_DRAWING_DIR)
    tasks = []
    for image_id, x, y in sorted(selected):
        gt = gtdict[image_id]
        dt = dtdict[image_id]
        crop = (x, y, 2048, 2048)
        file_name = os.path.join(settings.TEST_IMAGE_DIR, gt['file_name'])
        tasks.append((
            file_name,
            os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_{}_{}_{}_{}_gt.png'.format(image_id, *crop)),
            {
                'boxes': remove_outside(gt2array(gt, draw_ignore=True), crop),
                'crop': crop,
                'place': 'force',
            }
        ))
        tasks.append((
            file_name,
            os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_{}_{}_{}_{}_dt.png'.format(image_id, *crop)),
            {
                'boxes': remove_outside(dt2array(dt, gt, draw_ignore=True), crop),
                'crop': crop,
                'place': 'force',
            }
        ))
    common_tools.multithreaded(print_text, tasks, num_thread=cpu_count())


if __name__ == '__main__':
    main()
