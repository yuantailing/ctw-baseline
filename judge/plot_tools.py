# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.font_manager import FontProperties
from statistics_in_paper import get_chinese_ttf


def draw_bar(data, labels, width=None, xticks_font_fname=None, legend_kwargs=dict()):
    n = len(labels)
    m = len(data)
    if not width:
        width = 1. / (m + .6)
    off = 1.
    legend_bar = []
    legend_text = []
    for i, a in enumerate(data):
        for j, b in enumerate(a):
            assert n == len(b['data'])
            ind = [off + k + (i + (1 - m) / 2) * width for k in range(n)]
            bottom = [sum(d) for d in zip(*[c['data'] for c in a[j + 1:]])] or None
            p = plt.bar(ind, b['data'], width, bottom=bottom, color=b.get('color'))
            legend_bar.append(p[0])
            legend_text.append(b['legend'])
    ind = [off + i for i, label in enumerate(labels) if label is not None]
    labels = [label for label in labels if label is not None]
    font = FontProperties(fname=xticks_font_fname)
    plt.xticks(ind, labels, fontproperties=font, ha='center')
    plt.legend(legend_bar, legend_text, **legend_kwargs)


def print_text(in_file_name, out_file_name, obj):
    dpi = 72
    img = cv2.imread(in_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, crop = obj['boxes'], obj['crop']
    img = img[crop[1]:crop[1] + crop[3], crop[0]:crop[0]+crop[2]]

    with plt.style.context({
        'figure.subplot.left': 0.,
        'figure.subplot.right': 1.,
        'figure.subplot.bottom': 0.,
        'figure.subplot.top': 1.,
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
    }):
        font = FontProperties(fname=get_chinese_ttf())
        plt.figure(figsize=(crop[2] / dpi, crop[3] / dpi))
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        plt.imshow(img / 255)
        for o in boxes:
            bbox, text, color = o['bbox'], o['text'], o['color']
            if color.startswith('#'):
                color = color[1:]
                if 3 == len(color):
                    color = tuple(int(s, 16) / 15 for s in color)
            bbox = [bbox[0] - crop[0], bbox[1] - crop[1], bbox[2], bbox[3]]
            ax.add_patch(patches.Rectangle(bbox[:2], *bbox[2:], fill=False, color=color))
            ax.text(bbox[0], bbox[1], text, fontproperties=font, fontsize=8, color=color,
                    horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(out_file_name, dpi=dpi)
        plt.close()
