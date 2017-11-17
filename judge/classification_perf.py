# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import copy
import json
import matplotlib.pyplot as plt
import os
import plot_tools
import predictions2html
import settings
import six

from jinja2 import Template
from pythonapi import eval_tools
from six.moves import urllib


def get_chartjs():
    chartjs_file_path = os.path.join(settings.PRODUCTS_ROOT, 'Chart.min.js');
    if not os.path.isfile(chartjs_file_path):
        urllib.request.urlretrieve(
            'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.1/Chart.min.js',
            chartjs_file_path
        )
    with open(chartjs_file_path) as f:
        chartjs = f.read()
    return chartjs


def recall_print(recall, name):
    print(name, end=' ')
    for n, rc_n in sorted(recall['recalls'].items()):
        if recall['n'] > 0:
            print('{:4.1f}%'.format(rc_n / recall['n'] * 100), end=' ')
        else:
            print('{:4.1f}%'.format(0), end=' ')
    print('n={}'.format(recall['n']))


def main():
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        gt = f.read()
    all = list()
    for model in predictions2html.cfgs:
        with open(model['predictions_file_path']) as f:
            pr = f.read()
        report = eval_tools.classification_recall(
            gt, pr,
            settings.RECALL_N, settings.ATTRIBUTES, settings.SIZE_RANGES
        )
        assert 0 == report['error'], report['msg']
        all.append({
            'model_name': model['model_name'],
            'performance': report['performance'],
        })

    with open('explore_cls.template.html') as f:
        template = Template(f.read())
    jdata = copy.deepcopy(all)
    for model in jdata:
        for _, szperf in model['performance'].items():
            del szperf['texts']
    with codecs.open(settings.CLASSIFICATION_REPORT, 'w', 'utf-8') as f:
        f.write(template.render({
            'chartjs': get_chartjs(),
            'performance_all': json.dumps(jdata, sort_keys=True),
            'attributes': settings.ATTRIBUTES,
        }))

    def recall_empty():
        return {'recalls': {n: 0 for n in settings.RECALL_N}, 'n': 0}

    def recall_add(a, b):
        return {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in settings.RECALL_N}, 'n': a['n'] + b['n']}

    with open(settings.STAT_FREQUENCY) as f:
        frequency = json.load(f)
    freq_order = [o['text'] for o in frequency]
    for report_obj in all:
        print('[', report_obj['model_name'], ']')
        performance = report_obj['performance']
        for i, attr in zip(range(-1, len(settings.ATTRIBUTES)), ['__all__'] + settings.ATTRIBUTES):
            for szname, _ in sorted(settings.SIZE_RANGES):
                name = '{:12s} & {:12s}'.format(szname, attr)
                recall = recall_empty()
                for k, o in enumerate(performance[szname]['attributes']):
                    if i == -1 or int(k) & 2 ** i:
                        recall = recall_add(recall, o)
                recall_print(recall, name)
        for char in freq_order[:10]:
            recall_print(performance['all']['texts'][char], char)

    draw_by_models(all)
    for report_obj in all:
        draw_by_attrs(**report_obj)


def draw_by_models(all):
    def model_recall(attr_perfs):
        n = rc = 0
        for k, o in enumerate(attr_perfs):
            n += o['n']
            rc += o['recalls'][1]
        return 0. if n == 0 else rc / n
    data = [
        [
            {
                'legend': szname,
                'data': [model_recall(model['performance'][szname]['attributes']) for model in all],
            }
        ] for szname, _ in settings.SIZE_RANGES
    ]
    labels = [list(filter(lambda o: o['model_name'] == model['model_name'], predictions2html.cfgs))[0]['display_name'] for model in all]
    with plt.style.context({
        'figure.subplot.left': .10,
        'figure.subplot.right': .96,
        'figure.subplot.top': .96,
        'legend.loc': 'upper center',
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(6, 3))
        plt.ylim((0., 1.))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels, width=.18, legend_kwargs={'ncol': len(settings.SIZE_RANGES)})
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'cls_precision_by_model_size.pdf'))
        plt.close()


def draw_by_attrs(model_name, performance):
    def attr_recall(attr_perfs, attr_id):
        m = len(settings.ATTRIBUTES)
        n = rc = 0
        for k, o in enumerate(attr_perfs):
            if attr_id == -1 or (attr_id < m and 0 != int(k) & 2 ** attr_id) or (m <= attr_id and 0 == int(k) & 2 ** (attr_id - m)):
                n += o['n']
                rc += o['recalls'][1]
        return 0. if n == 0 else rc / n

    data = [
        [
            {
                'legend': szname,
                'data': [attr_recall(performance[szname]['attributes'], i) for i in range(-1, 2 * len(settings.ATTRIBUTES))],
            }
        ] for szname, _ in settings.SIZE_RANGES
    ]
    labels = ['all'] + settings.ATTRIBUTES + list(map('~{}'.format, settings.ATTRIBUTES))
    with plt.style.context({
        'figure.subplot.left': .05,
        'figure.subplot.right': .98,
        'figure.subplot.top': .96,
        'legend.loc': 'upper center',
        'pdf.fonttype': 42,
    }):
        plt.figure(figsize=(12, 3))
        plt.xlim((.3, .7 + len(labels)))
        plt.ylim((0., 1.))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels, width=.18, legend_kwargs={'ncol': len(settings.SIZE_RANGES)})
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'cls_precision_by_attr_size_{}.pdf'.format(model_name)))
        plt.close()


if __name__ == '__main__':
    main()
