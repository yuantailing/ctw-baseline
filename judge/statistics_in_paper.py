# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json
import settings
import six

from collections import defaultdict
from pythonapi import anno_tools


def main():
    assert six.PY3
    most_freq = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_char = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_uniq_char = defaultdict(lambda: {'trainval': 0, 'test': 0})
    num_image = {'trainval': 0, 'test': 0}
    sum_chinese = {'trainval': 0, 'test': 0}
    sum_not_chinese = {'trainval': 0, 'test': 0}
    sum_ignore = {'trainval': 0, 'test': 0}
    with open(settings.TRAIN) as f, open(settings.VAL) as f2:
        for line in f.read().splitlines() + f2.read().splitlines():
            anno = json.loads(line.strip())
            num = 0
            uniq = set()
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    most_freq[char['text']]['trainval'] += 1
                    num += 1
                    uniq.add(char['text'])
                    sum_chinese['trainval'] += 1
                else:
                    sum_not_chinese['trainval'] += 1
            assert 0 < len(uniq)
            num_char[num]['trainval'] += 1
            num_uniq_char[len(uniq)]['trainval'] += 1
            num_image['trainval'] += 1
            sum_ignore['trainval'] += len(anno['ignore'])
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        for line in f:
            gt = json.loads(line.strip())['ground_truth']
            num = 0
            uniq = set()
            for char in gt:
                most_freq[char['text']]['test'] += 1
                num += 1
                uniq.add(char['text'])
                sum_chinese['test'] += 1
            assert 0 < len(uniq)
            num_char[num]['test'] += 1
            num_uniq_char[num]['test'] += 1
            num_image['test'] += 1
    with open(settings.TEST_DETECTION_GT) as f:
        for line in f:
            anno = json.loads(line.strip())
            num = 0
            uniq = set()
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    most_freq[char['text']]['test'] += 1
                    num += 1
                    uniq.add(char['text'])
                    sum_chinese['test'] += 1
                else:
                    sum_not_chinese['test'] += 1
            assert 0 < len(uniq)
            num_char[num]['test'] += 1
            num_uniq_char[len(uniq)]['test'] += 1
            num_image['test'] += 1
            sum_ignore['test'] += len(anno['ignore'])
    most_freq = [{
        'text': k,
        'trainval': v['trainval'],
        'test': v['test'],
    } for k, v in most_freq.items()]
    most_freq.sort(key=lambda o: (-o['trainval'] - o['test'], o['text']))
    print('50_most_frequent_characters')
    for i, o in enumerate(most_freq[:50]):
        print(i + 1, o['text'], o['trainval'], o['test'])
    print('total_number_of_characters_in_each_image')
    for i in range(1, 61):
        print(i, num_char[i]['trainval'], num_char[i]['test'])
    print('number_of_different_characters_per_image')
    for i in range(1, 61):
        print(i, num_uniq_char[i]['trainval'], num_uniq_char[i]['test'])
    print('over_all')
    print('uniq_chinese', len(most_freq))
    print('num_image', num_image['trainval'], num_image['test'])
    print('sum_chinese', sum_chinese['trainval'], sum_chinese['test'])
    print('sum_not_chinese', sum_not_chinese['trainval'], sum_not_chinese['test'])
    print('sum_ignore', sum_ignore['trainval'], sum_ignore['test'])


if __name__ == '__main__':
    main()
