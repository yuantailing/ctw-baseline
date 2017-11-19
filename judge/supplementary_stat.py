# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import settings

from collections import defaultdict
from pythonapi import anno_tools


def main():
    all = defaultdict(int)
    n = 0
    with open(settings.TRAIN) as f1, open(settings.VAL) as f2:
        for line in f1.read().splitlines() + f2.read().splitlines():
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    k = 0
                    n += 1
                    for attr in char['attributes']:
                        k |= 2 ** settings.ATTRIBUTES.index(attr)
                    for szname, (lo, hi) in settings.SIZE_RANGES:
                        if lo <= max(char['adjusted_bbox'][2:]) < hi:
                            all['trainval', szname, k] += 1
    with open(settings.TEST_CLASSIFICATION_GT) as f2:
        for line in f2:
            gts = json.loads(line.strip())
            for char in gts['ground_truth']:
                k = 0
                for attr in char['attributes']:
                    k |= 2 ** settings.ATTRIBUTES.index(attr)
                for szname, (lo, hi) in settings.SIZE_RANGES:
                    if lo <= max(char['size']) < hi:
                        all['test', szname, k] += 1
    with open(settings.TEST_DETECTION_GT) as f:
        for line in f:
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    k = 0
                    for attr in char['attributes']:
                        k |= 2 ** settings.ATTRIBUTES.index(attr)
                    for szname, (lo, hi) in settings.SIZE_RANGES:
                        if lo <= max(char['adjusted_bbox'][2:]) < hi:
                            all['test', szname, k] += 1

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
            s = r'{}\!\!\!\!\! & \& & \!\!\!\!\!{}'.format(trans(i), trans(j))
            for szname, (lo, hi) in settings.SIZE_RANGES:
                n_train = n_test = 0
                for k, v in all.items():
                    if check(k[2], i) and check(k[2], j):
                        if ('trainval', szname) == k[:2]:
                            n_train += v
                        if ('test', szname) == k[:2]:
                            n_test += v
                s += ' & {} / {}'.format(n_train, '{:5d}'.format(n_test).replace(' ', r'\,\,\,'))
            s += r' \\'
            print(s)


if __name__ == '__main__':
    main()
