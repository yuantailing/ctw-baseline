# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import bisect
import codecs
import json
import matplotlib.pyplot as plt
import os
import plot_tools
import settings
import six
import threading

from collections import defaultdict
from pythonapi import anno_tools, common_tools
from six.moves import urllib


@common_tools.synchronized(threading.Lock())
def get_chinese_ttf():
    if not os.path.isdir(settings.PLOTS_DIR):
        os.makedirs(settings.PLOTS_DIR)
    chinese_ttf = os.path.join(settings.PRODUCTS_ROOT, 'SimHei.ttf')
    if not os.path.isfile(chinese_ttf) or 9751960 != os.path.getsize(chinese_ttf):
        url = 'http://fonts.cooltext.com/Downloader.aspx?ID=11120'
        print('please download {} to {}'.format(url, chinese_ttf))
        if os.path.isfile(chinese_ttf):
            os.unlink(chinese_ttf)
        urllib.request.urlretrieve('http://fonts.cooltext.com/Downloader.aspx?ID=11120',
                                   chinese_ttf)
    return chinese_ttf


def main():
    most_freq = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_char = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_uniq_char = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_image = {'trainval': 0, 'test': 0}
    sum_chinese = {'trainval': 0, 'test': 0}
    sum_not_chinese = {'trainval': 0, 'test': 0}
    sum_ignore = {'trainval': 0, 'test': 0}
    longsizes = {'trainval': list(), 'test': list()}
    attrs = {szname: {attr: 0 for attr in settings.ATTRIBUTES + ['__all__']} for szname, _ in settings.SIZE_RANGES}
    with open(settings.TRAIN) as f, open(settings.VAL) as f2:
        for line in f.read().splitlines() + f2.read().splitlines():
            anno = json.loads(line.strip())
            num = 0
            uniq = set()
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    most_freq[char['text']]['trainval'] += 1
                    num += 1
                    uniq.add(char['text'])
                    sum_chinese['trainval'] += 1
                    longsize = max(char['adjusted_bbox'][2], char['adjusted_bbox'][3])
                    longsizes['trainval'].append(longsize)
                    for szname, szrange in settings.SIZE_RANGES:
                        if szrange[0] <= longsize < szrange[1]:
                            for attr in char['attributes']:
                                attrs[szname][attr] += 1
                            attrs[szname]['__all__'] += 1
                else:
                    sum_not_chinese['trainval'] += 1
            assert 0 < len(uniq)
            num_char[num]['trainval'] += 1
            num_uniq_char[len(uniq)]['trainval'] += 1
            num_image['trainval'] += 1
            sum_ignore['trainval'] += len(anno['ignore'])
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        for line in f:
            gt = json.loads(line.strip())['ground_truth']
            num = 0
            uniq = set()
            for char in gt:
                most_freq[char['text']]['test'] += 1
                num += 1
                uniq.add(char['text'])
                sum_chinese['test'] += 1
                longsize = max(*char['size'])
                longsizes['test'].append(longsize)
                for szname, szrange in settings.SIZE_RANGES:
                    if szrange[0] <= longsize < szrange[1]:
                        for attr in char['attributes']:
                            attrs[szname][attr] += 1
                        attrs[szname]['__all__'] += 1
            assert 0 < len(uniq)
            num_char[num]['test'] += 1
            num_uniq_char[len(uniq)]['test'] += 1
            num_image['test'] += 1
    with open(settings.TEST_DETECTION_GT) as f:
        for line in f:
            anno = json.loads(line.strip())
            num = 0
            uniq = set()
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    most_freq[char['text']]['test'] += 1
                    num += 1
                    uniq.add(char['text'])
                    sum_chinese['test'] += 1
                    longsizes['test'].append(max(char['adjusted_bbox'][2], char['adjusted_bbox'][3]))
                    for szname, szrange in settings.SIZE_RANGES:
                        if szrange[0] <= longsize < szrange[1]:
                            for attr in char['attributes']:
                                attrs[szname][attr] += 1
                            attrs[szname]['__all__'] += 1
                else:
                    sum_not_chinese['test'] += 1
            assert 0 < len(uniq)
            num_char[num]['test'] += 1
            num_uniq_char[len(uniq)]['test'] += 1
            num_image['test'] += 1
            sum_ignore['test'] += len(anno['ignore'])
    most_freq = [{
        'text': k,
        'trainval': v['trainval'],
        'test': v['test'],
    } for k, v in most_freq.items()]
    most_freq.sort(key=lambda o: (-o['trainval'] - o['test'], o['text']))
    print('10_most_frequent_characters')
    for i, o in enumerate(most_freq[:10]):
        print(i + 1, o['text'], o['trainval'], o['test'])
    print('over_all')
    print('uniq_chinese', len(most_freq))
    print('num_image', num_image['trainval'], num_image['test'])
    print('sum_chinese', sum_chinese['trainval'], sum_chinese['test'])
    print('sum_not_chinese', sum_not_chinese['trainval'], sum_not_chinese['test'])
    print('sum_ignore', sum_ignore['trainval'], sum_ignore['test'])

    with codecs.open(settings.STAT_FREQUENCY, 'w', 'utf-8') as f:
        json.dump(most_freq, f, ensure_ascii=False, indent=2)

    # most_freq
    meta = most_freq[:50]
    data = [
        [
            {
                'legend': 'training set',
                'data': [o['trainval'] for o in meta],
            }, {
                'legend': 'testing set',
                'data': [o['test'] for o in meta],
            },
        ],
    ]
    labels = [o['text'] for o in meta]
    with plt.style.context({
        'figure.subplot.left': .06,
        'figure.subplot.right': .98,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(10, 3))
        plt.xlim((0, len(labels) + 1))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels, xticks_font_fname=get_chinese_ttf())
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'stat_most_freq.pdf'))
        plt.close()

    # num_char
    meta = [num_char[i] for i in range(1, 61)]
    data = [
        [
            {
                'legend': 'training set',
                'data': [o['trainval'] for o in meta],
            }, {
                'legend': 'testing set',
                'data': [o['test'] for o in meta],
            },
        ],
    ]
    labels = [i + 1 if (i + 1) % 10 == 0 else None for i, _ in enumerate(meta)]
    with plt.style.context({
        'figure.subplot.left': .14,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .16,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(5, 3))
        plt.xlim((0, len(labels) + 1))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels)
        plt.xlabel('number of character instances')
        plt.ylabel('number of images')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'stat_num_char.pdf'))
        plt.close()

    # num_uniq_char
    meta = [num_uniq_char[i] for i in range(1, 61)]
    data = [
        [
            {
                'legend': 'training set',
                'data': [o['trainval'] for o in meta],
            }, {
                'legend': 'testing set',
                'data': [o['test'] for o in meta],
            },
        ],
    ]
    labels = [i + 1 if (i + 1) % 10 == 0 else None for i, _ in enumerate(meta)]
    with plt.style.context({
        'figure.subplot.left': .14,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .16,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(5, 3))
        plt.xlim((0, len(labels) + 1))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels)
        plt.xlabel('number of character categories')
        plt.ylabel('number of images')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'stat_num_uniq_char.pdf'))
        plt.close()

    # instance size
    longsizes['trainval'].sort()
    longsizes['test'].sort()
    ranges = list(range(0, 65, 8))
    data = [
        [
            {
                'legend': 'training set',
                'data': [bisect.bisect_left(longsizes['trainval'], hi) - bisect.bisect_left(longsizes['trainval'], lo) for lo, hi in zip(ranges, ranges[1:] + [float('inf')])],
            }, {
                'legend': 'testing set',
                'data': [bisect.bisect_left(longsizes['test'], hi) - bisect.bisect_left(longsizes['test'], lo) for lo, hi in zip(ranges, ranges[1:] + [float('inf')])],
            },
        ],
    ]
    labels = ['{}-{}'.format(lo, hi) for lo, hi in zip(ranges, ranges[1:] + [''])]
    with plt.style.context({
        'figure.subplot.left': .12,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .10,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(6, 3))
        plt.xlim((0, len(labels) + 1))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels)
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'stat_instance_size.pdf'))
        plt.close()

    # attributes percentage
    data = [
        [
            {
                'legend': szname,
                'data': [attrs[szname][attr] / attrs[szname]['__all__'] * 100 for attr in settings.ATTRIBUTES],
            }
        ] for szname, szrange in settings.SIZE_RANGES
    ]
    labels = settings.ATTRIBUTES
    with plt.style.context({
        'figure.subplot.left': .12,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .10,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(6, 3))
        plt.xlim((.3, .7 + len(labels)))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
        plot_tools.draw_bar(data, labels, width=.18)
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'stat_attributes.pdf'))
        plt.close()


if __name__ == '__main__':
    main()
