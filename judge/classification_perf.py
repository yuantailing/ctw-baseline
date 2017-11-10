# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import copy
import json
import os
import predictions2html
import settings
import six
import sys

from jinja2 import Template
from pythonapi import eval_tools
from six.moves import urllib


def recall_print(recall, name):
    print(name, end=' ')
    for n, rc_n in sorted(recall['recalls'].items()):
        if recall['n'] > 0:
            print('{:5.2f}%'.format(rc_n / recall['n'] * 100), end=' ')
        else:
            print('{:5.2f}%'.format(0), end=' ')
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
            settings.RECALL_N, settings.PROPERTIES, settings.SIZE_RANGES
        )
        assert 0 == report['error'], report['msg']
        all.append({
            'model_name': model['model_name'],
            'performance': report['performance'],
        })

    chartjs_file_path = os.path.join(settings.PRODUCTS_ROOT, 'Chart.min.js');
    if not os.path.isfile(chartjs_file_path):
        urllib.request.urlretrieve(
            'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.1/Chart.min.js',
            chartjs_file_path
        )
    with open(chartjs_file_path) as f:
        chartjs = f.read()
    with open('explore_cls.template.html') as f:
        template = Template(f.read())
    jdata = copy.deepcopy(all)
    for model in jdata:
        for _, szperf in model['performance'].items():
            del szperf['texts']
    with codecs.open(settings.CLASSIFICATION_REPORT, 'w', 'utf-8') as f:
        f.write(template.render({
            'chartjs': chartjs,
            'performance_all': json.dumps(jdata, sort_keys=True),
            'properties': settings.PROPERTIES,
        }))

    def recall_empty():
        return {'recalls': {n: 0 for n in settings.RECALL_N}, 'n': 0}

    def recall_add(a, b):
        return {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in settings.RECALL_N}, 'n': a['n'] + b['n']}

    for report_obj in all:
        print('[', report_obj['model_name'], ']')
        performance = report_obj['performance']
        for szname, _ in sorted(settings.SIZE_RANGES):
            name = '{:12s} & {:12s}'.format(szname, '__all__')
            recall = recall_empty()
            for k, v in performance[szname]['properties'].items():
                recall = recall_add(recall, v)
            recall_print(recall, name)
        for i, prop in enumerate(settings.PROPERTIES):
            for szname, _ in sorted(settings.SIZE_RANGES):
                name = '{:12s} & {:12s}'.format(szname, prop)
                recall = recall_empty()
                for k, v in performance[szname]['properties'].items():
                    if int(k) & 2 ** i:
                        recall = recall_add(recall, v)
                recall_print(recall, name)
        for char, recall in sorted(performance['__all__']['texts'].items(), key=lambda o: -o[1]['n'])[:10]:
            recall_print(recall, char)


if __name__ == '__main__':
    main()
