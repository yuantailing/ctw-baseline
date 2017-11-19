# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import matplotlib.pyplot as plt
import os
import plot_tools
import settings

from pythonapi import anno_tools


def plt_print_text(*args):
    print('plot_tools.print_text', *args[:-1])
    with plt.style.context({
        'pdf.fonttype': 42,
    }):
        plot_tools.print_text(*args)
plt_print_text.concurrent = False

print_text = plt_print_text


def main():
    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    lines = []
    with open(settings.TRAIN) as f:
        lines += f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()
    with open(settings.TEST_DETECTION_GT) as f:
        lines += f.read().splitlines()

    def gt2array(gt):
        color = '#0f0'
        a = list()
        for char in anno_tools.each_char(gt):
            if char['is_chinese']:
                a.append({'polygon': char['polygon'], 'text': '', 'color': color, 'fontsize': 10})
        return a

    selected = [
        '1026027',
    ]

    if not os.path.isdir(settings.PRINTTEXT_DRAWING_DIR):
        os.makedirs(settings.PRINTTEXT_DRAWING_DIR)
    tasks = []
    for image_id in selected:
        i = [o['image_id'] for o in data_list['train'] + data_list['val'] + data_list['test_det']].index(image_id)
        gt = json.loads(lines[i])
        crop = (0, 0, gt['width'], gt['height'])
        file_name = os.path.join(settings.TRAINVAL_IMAGE_DIR if i < len(data_list['train'] + data_list['val']) else settings.TEST_IMAGE_DIR, gt['file_name'])
        output = os.path.join(settings.PRINTTEXT_DRAWING_DIR, 'gt_{}.pdf'.format(image_id))
        print_text(file_name, output, {
            'boxes': gt2array(gt),
            'crop': crop,
        })


if __name__ == '__main__':
    main()
