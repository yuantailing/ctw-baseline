# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


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
