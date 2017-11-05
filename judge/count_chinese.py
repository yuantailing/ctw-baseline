# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import settings

from collections import defaultdict
from pythonapi import anno_tools


def main():
    counts = defaultdict(lambda: {'trainval': 0, 'test_cls': 0, 'test_det': 0})
    with open(settings.TRAIN) as f, open(settings.VAL) as f2:
        for line in f.read().splitlines() + f2.read().splitlines():
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    counts[char['text']]['trainval'] += 1
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        for line in f:
            gt = json.loads(line.strip())['ground_truth']
            for char in gt:
                counts[char['text']]['test_cls'] += 1
    with open(settings.TEST_DETECTION_GT) as f:
        for line in f:
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    counts[char['text']]['test_det'] += 1

    counts = [{
        'text': k,
        'trainval': v['trainval'],
        'test_cls': v['test_cls'],
        'test_det': v['test_det'],
    } for k, v in counts.items()]
    counts.sort(key=lambda o: (-o['trainval'] - o['test_cls'] - o['test_det'], o['text']))
    for i, o in enumerate(counts):
        o['id'] = i
    with codecs.open(settings.COUNT_CHINESE, 'w', 'utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
