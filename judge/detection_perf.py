# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import matplotlib.pyplot as plt
import numpy as np
import settings

from pythonapi import eval_tools


def main(dt_file_path):
    with open(settings.TEST_DETECTION_GT) as f:
        gt = f.read()
    with open(dt_file_path) as f:
        dt = f.read()
    report = eval_tools.detection_mAP(
        gt, dt,
        settings.PROPERTIES, settings.SIZE_RANGES, settings.MAX_DET_PER_IMAGE, settings.IOU_THRESH
    )
    assert 0 == report['error'], report['msg']
    with codecs.open(settings.DETECTION_REPORT, 'w', 'utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    show(report)


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
        for prop, recall in sorted(stat['properties'].items()):
            r = 0. if recall['n'] == 0 else recall['recall'] / recall['n']
            print('{:13s}'.format(prop), 'n', '=', '{:6d}'.format(recall['n']), ',', 'recall', '=', percentage(r), '(at most {} guesses per image)'.format(settings.MAX_DET_PER_IMAGE))
        print()
        y = [1.] + stat['curve'] + [0.] * (stat['n'] - len(stat['curve']))
        x = np.linspace(0, 1, len(y))
        plt.plot(x, y, label=szname.replace('_', ''))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main('../detection/products/detections.jsonl')
