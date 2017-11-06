# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import predictions2html
import settings
import sys

from jinja2 import Template
from pythonapi import eval_tools


def recall_print(recall, name):
    print(name, end=' ')
    for n, rc_n in sorted(recall['recalls'].items()):
        if recall['n'] > 0:
            print('{:5.2f}%'.format(rc_n / recall['n'] * 100), end=' ')
        else:
            print('{:5.2f}%'.format(0), end=' ')
    print('n={}'.format(recall['n']))


def main(model_names):
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        gt = f.read()
    all = list()
    for model_name in model_names:
        model = list(filter(lambda o: o['model_name'] == model_name, predictions2html.cfgs))[0]
        with open(model['predictions_file_path']) as f:
            pr = f.read()
        report = eval_tools.classification_recall(
            gt, pr,
            settings.RECALL_N, settings.PROPERTIES, settings.SIZE_RANGES
        )
        assert 0 == report['error'], report['msg']
        all.append({
            'model_name': model_name,
            'report': report,
        })
    with open('explore_cls.template.html') as f:
        template = Template(f.read())
    with codecs.open(settings.CLASSIFICATION_REPORT, 'w', 'utf-8') as f:
        f.write(template.render({
            'report_all': json.dumps(all, ensure_ascii=False, sort_keys=True, indent='\t'),
            'properties': settings.PROPERTIES,
        }))

    def recall_empty():
        return {'recalls': {n: 0 for n in settings.RECALL_N}, 'n': 0}

    def recall_add(a, b):
        return {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in settings.RECALL_N}, 'n': a['n'] + b['n']}

    for report_obj in all:
        print('[', report_obj['model_name'], ']')
        report = report_obj['report']
        for szname, _ in sorted(settings.SIZE_RANGES):
            name = '{:12s} & {:12s}'.format(szname, '__all__')
            recall = recall_empty()
            for k, v in report['performance'][szname].items():
                recall = recall_add(recall, v)
            recall_print(recall, name)
        for i, prop in enumerate(settings.PROPERTIES):
            for szname, _ in sorted(settings.SIZE_RANGES):
                name = '{:12s} & {:12s}'.format(szname, prop)
                recall = recall_empty()
                for k, v in report['performance'][szname].items():
                    if int(k) & 2 ** i:
                        recall = recall_add(recall, v)
                recall_print(recall, name)
        for char, recall in sorted(report['group_by_characters'].items(), key=lambda o: -o[1]['n'])[:10]:
            recall_print(recall, char)


if __name__ == '__main__':
    assert 1 < len(sys.argv)
    main(sys.argv[1:])
