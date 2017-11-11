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
        settings.PROPERTIES, settings.SIZE_RANGES, settings.MAX_DET_PER_IMAGE, settings.IOU_THRESH,
        echo=True
    )
    assert 0 == report['error'], report['msg']
    with codecs.open(settings.DETECTION_REPORT, 'w', 'utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    html_explore(report)
    show(report)
    draw(report)


def html_explore(report):
    jdata = [{
        'model_name': 'YOLO_v2',
        'performance': {
            szname: {
                'properties': [
                    {'n': o['n'], 'recalls': {1: o['recall']}} for o in szprop['properties']
                ],
            } for szname, szprop in report['performance'].items()
        },
    }]
    with open('explore_cls.template.html') as f:
        template = Template(f.read())
    with codecs.open(settings.DETECTION_EXPLORE, 'w', 'utf-8') as f:
        f.write(template.render({
            'title': 'Explore detection performance',
            'chartjs': get_chartjs(),
            'performance_all': json.dumps(jdata, sort_keys=True),
            'properties': settings.PROPERTIES,
        }))


def show(report):
    def percentage(x, digit=2):
        fmt = {
            1: '{:4.1f}%',
            2: '{:5.2f}%',
        }
        return fmt[digit].format(x * 100)
    performance = report['performance']
    for szname, stat in sorted(performance.items()):
        print(szname)
        for k in ('n', 'mAP', 'AP'):
            x = stat[k]
            if isinstance(x, float):
                x = percentage(x)
            print('{:>4s}'.format(k), '=', x)
        for i, prop in zip(range(-1, len(settings.PROPERTIES)), ['__all__'] + settings.PROPERTIES):
            n = 0
            rc = 0
            for k, o in enumerate(performance[szname]['properties']):
                if i == -1 or int(k) & 2 ** i:
                    n += o['n']
                    rc += o['recall']
            r = 0. if n == 0 else rc / n
            print('{:13s}'.format(prop), 'n', '=', '{:6d}'.format(n), ',', 'recall', '=', percentage(r), '(at most {} guesses per image)'.format(settings.MAX_DET_PER_IMAGE))
        print()


def draw(report):
    def prop_recall(prop_perfs, prop_id):
        n = rc = 0
        for k, o in enumerate(prop_perfs):
            if prop_id == -1 or int(k) & 2 ** prop_id:
                n += o['n']
                rc += o['recall']
        return 0. if n == 0 else rc / n

    data = [
        [
            {
                'legend': szname,
                'data': [prop_recall(report['performance'][szname]['properties'], i) for i in range(-1, len(settings.PROPERTIES))],
            }
        ] for szname, _ in settings.SIZE_RANGES
    ]
    labels = ['all'] + settings.PROPERTIES
    with plt.style.context({
        'figure.subplot.left': .06,
        'figure.subplot.right': .84,
        'figure.subplot.top': .96,
    }):
        plt.figure(figsize=(10, 3))
        plt.ylim((0., 1.))
        plt.grid(which='major', axis='y', linestyle='dotted')
        plot_tools.draw_bar(data, labels, width=.18, legend_bbox_to_anchor=(1., 1.))
        plt.ylabel('Recall')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'det_recall_by_props_size.svg'))
        plt.close()

    with plt.style.context({
        'figure.subplot.left': .14,
        'figure.subplot.right': .96,
        'figure.subplot.bottom': .12,
        'figure.subplot.top': .94,
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
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(settings.PLOTS_DIR, 'det_AP_curve.svg'))
        plt.close()


if __name__ == '__main__':
    main('../detection/products/detections.jsonl')
