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
                a.append({'polygon': char['polygon'], 'text': char['text'], 'color': color, 'fontsize': 10})
        for char in gt['ignore']:
            a.append({'polygon': char['polygon'], 'text': '', 'color': '#ff0', 'fontsize': 10})
        return a

    selected = [
        ('0000507', 0, 0, 2048, 2048),
        ('1023899', 0, 0, 2048, 2048),
        ('1031755', 0, 0, 2048, 2048),
        ('1044721', 0, 0, 2048, 2048),
        ('1046905', 0, 0, 2048, 2048),
        ('2000215', 0, 0, 2048, 2048),
        ('2004154', 0, 0, 2048, 2048),
        ('2005679', 0, 0, 2048, 2048),
        ('2024003', 0, 0, 2048, 2048),
        ('3005669', 0, 0, 2048, 2048),
        ('3029319', 0, 0, 2048, 2048),
        ('3040629', 0, 0, 2048, 2048),
        ('3001838', 0, 650, 700, 550),
        ('1041797', 530, 740, 700, 550),
    ]

    if not os.path.isdir(settings.PRINTTEXT_DRAWING_DIR):
        os.makedirs(settings.PRINTTEXT_DRAWING_DIR)
    tasks = []
    for image_id, x, y, w, h in selected:
        i = [o['image_id'] for o in data_list['train'] + data_list['val'] + data_list['test_det']].index(image_id)
        gt = json.loads(lines[i])
        crop = (x, y, w, h)
        file_name = os.path.join(settings.TRAINVAL_IMAGE_DIR if i < len(data_list['train'] + data_list['val']) else settings.TEST_IMAGE_DIR, gt['file_name'])
        output = os.path.join(settings.PRINTTEXT_DRAWING_DIR, 'gt_{}_{}_{}_{}_{}.pdf'.format(image_id, x, y, w, h))
        print_text(file_name, output, {
            'boxes': gt2array(gt),
            'crop': crop,
        })


if __name__ == '__main__':
    main()
