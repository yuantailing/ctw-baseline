# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import random
import settings
import six


def main():
    assert six.PY3
    random.seed(1)

    with open(settings.DETECTION_REPORT) as f:
        report = json.load(f)
    performance = report['performance']

    with open(settings.STAT_FREQUENCY) as f:
        frequency = json.load(f)
    freq_order = [o['text'] for o in frequency]
    with open('../classification/products/cates.json') as f:
        cates = json.load(f)
    cates = [(c['text'], [], c['cate_id']) for c in sorted(random.sample(cates[10:1000], 50), key=lambda o: -o['trainval'])]
    cates.sort(key=lambda t: freq_order.index(t[0]))
    for no, (text, a, cate_id) in enumerate(cates):
        s = '{} & '.format(no + 1) + r'\begin{minipage}{3.5mm} \includegraphics[width=\linewidth]{figure/texts/' + '0_{}.png'.format(freq_order.index(text)) + r'} \end{minipage}'
        for szname, _ in settings.SIZE_RANGES:
            APn = performance[szname]['texts'].get(text)
            if APn is None:
                s += ' & -'
            else:
                AP = round(APn['AP'] * 100, 1)
                s += ' & ' + '{:.1f}'.format(AP)
        s += ' & {}'.format(performance['all']['texts'].get(text, {'n': 0})['n'])
        s += r' & {} \\'.format(list(filter(lambda o: o['text'] == text, frequency))[0]['trainval'])
        print(s)

    def check(k, attr_id):
        if attr_id < len(settings.ATTRIBUTES):
            return int(k) & 2 ** attr_id
        else:
            return 0 == int(k) & 2 ** (attr_id - len(settings.ATTRIBUTES))
    def trans(attr_id):
        if attr_id < len(settings.ATTRIBUTES):
            return settings.ATTRIBUTES[attr_id]
        else:
            return r'$\sim${}'.format(settings.ATTRIBUTES[attr_id - len(settings.ATTRIBUTES)])

    for i in range(2 * len(settings.ATTRIBUTES) - 1):
        for j in range(i + 1, 2 * len(settings.ATTRIBUTES)):
            if j == i + len(settings.ATTRIBUTES):
                continue
            s = r'{} & \& & {}'.format(trans(i), trans(j))
            for szname, _ in settings.SIZE_RANGES:
                n = rc = 0
                for k, o in enumerate(performance[szname]['attributes']):
                    if check(k, i) and check(k, j):
                        n += o['n']
                        rc += o['recall']
                if n > 0:
                    rc_rate = round(rc / n * 100, 1)
                    s += ' & {:.1f}'.format(rc_rate)
                else:
                    s += ' & -'
            s += r' \\'
            print(s)


if __name__ == '__main__':
    main()
