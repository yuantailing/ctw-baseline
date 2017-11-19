# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_tools
import settings
import sys

from classification_perf import get_chartjs
from jinja2 import Template
from pythonapi import eval_tools


def main(dt_file_path):
    with open(settings.TEST_DETECTION_GT) as f:
        gt = f.read()
    with open(dt_file_path) as f:
        dt = f.read()
    report = eval_tools.detection_mAP(
        gt, dt,
        settings.ATTRIBUTES, settings.SIZE_RANGES, settings.MAX_DET_PER_IMAGE, settings.IOU_THRESH,
        proposal=proposal,
        echo=True
    )
    assert 0 == report['error'], report['msg']
    with codecs.open(settings.PROPOSAL_REPORT if proposal else settings.DETECTION_REPORT, 'w', 'utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    html_explore(report)
    show(report)
    draw(report)


def html_explore(report):
    jdata = [{
        'model_name': 'YOLO_v2',
        'performance': {
            szname: {
                'attributes': [
                    {'n': o['n'], 'recalls': {1: o['recall']}} for o in szattr['attributes']
                ],
            } for szname, szattr in report['performance'].items()
        },
    }]
    with open('explore_cls.template.html') as f:
        template = Template(f.read())
    with codecs.open(settings.PROPOSAL_EXPLORE if proposal else settings.DETECTION_EXPLORE, 'w', 'utf-8') as f:
        f.write(template.render({
            'title': 'Explore detection performance',
            'chartjs': get_chartjs(),
            'performance_all': json.dumps(jdata, sort_keys=True),
            'attributes': settings.ATTRIBUTES,
        }))


def show(report):
    def percentage(x, digit=1):
        fmt = {
            1: '{:4.1f}%',
            2: '{:5.2f}%',
        }
        return fmt[digit].format(x * 100)
    with open(settings.STAT_FREQUENCY) as f:
        frequency = json.load(f)
    freq_order = [o['text'] for o in frequency]
    performance = report['performance']
    for szname, stat in sorted(performance.items()):
        print(szname)
        for k in ('n', 'mAP', 'AP'):
            x = stat[k]
            if isinstance(x, float):
                x = percentage(x)
            print('{:>4s}'.format(k), '=', x)
        for i, attr in zip(range(-1, len(settings.ATTRIBUTES)), ['__all__'] + settings.ATTRIBUTES):
            n = 0
            rc = 0
            for k, o in enumerate(performance[szname]['attributes']):
                if i == -1 or int(k) & 2 ** i:
                    n += o['n']
                    rc += o['recall']
            r = 0. if n == 0 else rc / n
            print('{:13s}'.format(attr), 'n', '=', '{:6d}'.format(n), ',', 'recall', '=', percentage(r))
        for char in freq_order[:10]:
            print(char, percentage(performance[szname]['texts'].get(char, {'AP': 0.})['AP']))
        print()


def draw(report):
    def attr_recall(attr_perfs, attr_id):
        m = len(settings.ATTRIBUTES)
        n = rc = 0
        for k, o in enumerate(attr_perfs):
            if attr_id == -1 or (attr_id < m and 0 != int(k) & 2 ** attr_id) or (m <= attr_id and 0 == int(k) & 2 ** (attr_id - m)):
                n += o['n']
                rc += o['recall']
        return 0. if n == 0 else rc / n

    data = [
        [
            {
                'legend': szname,
                'data': [attr_recall(report['performance'][szname]['attributes'], i) for i in range(-1, 2 * len(settings.ATTRIBUTES))],
            }
        ] for szname, _ in settings.SIZE_RANGES
    ]
    labels = ['all'] + settings.ATTRIBUTES + list(map('~{}'.format, settings.ATTRIBUTES))
    with plt.style.context({
        'figure.subplot.left': .05,
        'figure.subplot.right': .98,
        'figure.subplot.top': .96,
        'pdf.fonttype': 42,
        'legend.loc': 'upper center',
    }):
        plt.figure(figsize=(12, 3))
        plt.xlim((.3, .7 + len(labels)))
        plt.ylim((0., 1.))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels, width=.18, legend_kwargs={'ncol': len(settings.SIZE_RANGES)})
        plt.ylabel('recall')
        plt.savefig(os.path.join(settings.PLOTS_DIR, ('pro' if proposal else 'det') + '_recall_by_attr_size.pdf'))
        plt.close()

    with plt.style.context({
        'figure.subplot.left': .14,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .12,
        'figure.subplot.top': .94,
        'pdf.fonttype': 42,
        'legend.loc': 'upper right',
    }):
        plt.figure(figsize=(5.5, 5.5))
        plt.xlim((0., 1.))
        plt.ylim((0., 1.))
        plt.grid(which='major', axis='both', linestyle='dotted')
        for szname, stat in sorted(report['performance'].items()):
            y = [1.] + stat['curve'] + [0.] * (stat['n'] - len(stat['curve']))
            x = np.linspace(0, 1, len(y))
            plt.plot(x, y, label=szname)
        plt.legend()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.savefig(os.path.join(settings.PLOTS_DIR, ('pro' if proposal else 'det') + '_AP_curve.pdf'))
        plt.close()


if __name__ == '__main__':
    proposal = 'proposal' in sys.argv[1:]
    main('../detection/products/proposals.jsonl' if proposal else '../detection/products/detections.jsonl')
