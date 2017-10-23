# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import predictions2html
import settings
import sys

from pythonapi import eval_tools


def recall_print(recall, name):
    print(name, end=' ')
    for n, rc_n in sorted(recall['recalls'].items()):
        if recall['n'] > 0:
            print('{:5.2f}%'.format(rc_n / recall['n'] * 100), end=' ')
        else:
            print('{:5.2f}%'.format(0), end=' ')
    print('n={}'.format(recall['n']))


def main(model_name):
    model = list(filter(lambda o: o['model_name'] == model_name, predictions2html.cfgs))[0]
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        gt = f.read()
    with open(model['predictions_file_path']) as f:
        pr = f.read()
    report = eval_tools.classification_recall(gt, pr,
        settings.RECALL_N, settings.PROPERTIES, settings.SIZE_RANGES)
    assert 0 == report['error'], report['msg']
    for prop in ['__all__'] + settings.PROPERTIES + ['~{}'.format(prop) for prop in settings.PROPERTIES]:
        for szname, _ in sorted(settings.SIZE_RANGES):
            name = '{:12s} & {:12s}'.format(szname, prop)
            recall = report['statistics'][szname][prop]
            recall_print(recall, name)
    for char, recall in sorted(report['group_by_characters'].items(), key=lambda o: -o[1]['n'])[:10]:
        recall_print(recall, char)


if __name__ == '__main__':
    main(sys.argv[1])
