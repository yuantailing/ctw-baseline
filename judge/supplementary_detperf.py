# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import settings


def main():
    with open(settings.DETECTION_REPORT) as f:
        report = json.load(f)
    performance = report['performance']

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
                    s += ' & {:.1f}'.format(rc / n * 100)
                else:
                    s += ' & -'
            s += r' \\'
            print(s)


if __name__ == '__main__':
    main()
