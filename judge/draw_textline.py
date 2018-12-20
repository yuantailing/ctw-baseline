# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_tools
import settings

from pythonapi import common_tools
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def plt_print_text(*args):
    print('plot_tools.print_text', *args[:-1])
    with plt.style.context({
        'pdf.fonttype': 42,
    }):
        plot_tools.print_text(*args)
plt_print_text.concurrent = False

print_text = plt_print_text


def work(selected, ext):
    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    with open(settings.TEST_DETECTION_GT) as f:
        gts = f.read().splitlines()
    with open('../detection/products/textlines.jsonl') as f:
        dts = f.read().splitlines()
    assert len(gts) == len(dts)

    def gt_helper(line):
        line = [char for char in line if char['is_chinese']]
        if not line:
            return None
        points = np.array([p for char in line for p in char['polygon']])
        hull = ConvexHull(points)
        polygon = [points[i].tolist() for i in hull.vertices]
        text = common_tools.reduce_sum(char['text'] for char in line)
        return Polygon(polygon), text

    def gt2array(gt, draw_ignore):
        color = '#0f0'
        color_ignore = '#ff0'
        a = list()
        for line in gt['annotations']:
            polygon, text = gt_helper(line)
            x, y = polygon.exterior.coords.xy
            polygon = list(zip(x, y))
            a.append({'polygon': polygon, 'text': text, 'color': color})
        if draw_ignore:
            for char in gt['ignore']:
                a.append({'bbox': char['bbox'], 'text': '', 'color': color_ignore})
        return a

    def dt2array(dtobj):
        dt = dtobj['textlines']
        a = list()
        for line in dt:
            polygon, text = line['polygon'], line['text']
            a.append({'polygon': polygon, 'text': text, 'color': '#f00'})
        return a

    if selected is None:
        selected = [(o['image_id'], 0, 0, 2048, 2048) for i, o in enumerate(data_list['test_det']) if i % 200 == 0]
    draw_gt = True

    if not os.path.isdir(settings.PRINTTEXT_DRAWING_DIR):
        os.makedirs(settings.PRINTTEXT_DRAWING_DIR)
    tasks = []
    for image_id, x, y, w, h in sorted(selected):
        i = [o['image_id'] for o in data_list['test_det']].index(image_id)
        gt = json.loads(gts[i])
        dt = json.loads(dts[i])
        crop = (x, y, w, h)
        file_name = os.path.join(settings.TEST_IMAGE_DIR, gt['file_name'])
        if draw_gt:
            tasks.append((
                file_name,
                os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_{}_{}_{}_{}_textline_gt.{}'.format(image_id, crop[0], crop[1], crop[2], crop[3], ext)),
                {
                    'boxes': gt2array(gt, draw_ignore=True),
                    'crop': crop,
                    'place': 'force',
                }
            ))
        tasks.append((
            file_name,
            os.path.join(settings.PRINTTEXT_DRAWING_DIR, '{}_{}_{}_{}_{}_textline_dt.{}'.format(image_id, crop[0], crop[1], crop[2], crop[3], ext)),
            {
                'boxes': dt2array(dt),
                'crop': crop,
                'place': 'force',
            }
        ))
    if print_text.concurrent:
        common_tools.multithreaded(print_text, tasks, num_thread=cpu_count())
    else:
        for task in tasks:
            print_text(*task)

def main():
    selected = [
        ('1009884', 1380, 675, 600, 500),
        ('2029053', 1300, 920, 600, 500),
    ]
    work(selected, 'pdf')


if __name__ == '__main__':
    main()
    # work(None, 'jpg')
