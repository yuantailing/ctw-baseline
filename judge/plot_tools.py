# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import operator

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
    dpi = 100
    img = cv2.imread(in_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, crop = obj['boxes'], obj['crop']
    img = img[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]]

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
        plt.figure(figsize=(crop[2] / dpi, crop[3] / dpi), dpi=dpi)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        plt.imshow(img)
        for o in boxes:
            bbox, polygon, text, color, fontsize = o.get('bbox'), o.get('polygon'), o['text'], o['color'], o.get('fontsize', 10)
            assert operator.xor(bbox is None, polygon is None)
            if color.startswith('#'):
                color = color[1:]
                if 3 == len(color):
                    color = tuple(int(s, 16) / 15 for s in color)
                else:
                    assert 6 == len(color)
                    color = tuple(int(color[i * 2:(i + 1) * 2], 16) / 255 for i in range(0, len(color), 2))
            if bbox is not None:
                bbox = [bbox[0] - crop[0], bbox[1] - crop[1], bbox[2], bbox[3]]
                if bbox[0] + bbox[2] < 0 or bbox[0] >= crop[2] or bbox[1] + bbox[3] < 0 or bbox[1] >= crop[3]:
                    continue
                ax.add_patch(patches.Rectangle(bbox[:2], *bbox[2:], fill=False, color=color))
                text_base = (bbox[0], bbox[1])
            if polygon is not None:
                polygon = [(xy[0] - crop[0], xy[1] - crop[1]) for xy in polygon]
                xy = list(zip(*polygon))
                if max(xy[0]) < 0 or min(xy[0]) >= crop[2] or max(xy[1]) < 0 or min(xy[1]) >= crop[3]:
                    continue
                def f(x, y):
                    return x + y
                fmin = f(*polygon[0])
                text_base = polygon[0]
                for xy in polygon[1:]:
                    if f(*xy) < fmin:
                        fmin = f(*xy)
                        text_base = xy
                ax.add_patch(patches.Polygon(polygon, fill=False, color=color))
            if text:
                if text_base[0] < 0 or text_base[0] >= crop[2] or text_base[1] < 0 or text_base[1] >= crop[3]:  # case test outside of crop
                    continue
                ax.text(text_base[0], text_base[1], text, fontproperties=font, fontsize=fontsize, color=color,
                        horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(out_file_name)
        plt.close()
