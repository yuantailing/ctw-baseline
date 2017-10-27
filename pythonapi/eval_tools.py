# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import six

from collections import defaultdict


def classification_recall(ground_truth, prediction, recall_n, properties, size_ranges):
    def error(s):
        return {'error': 1, 'msg': s}
    recall_empty = lambda: {'recalls': {n: 0 for n in recall_n}, 'n': 0}
    recall_add = lambda a, b: {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in recall_n}, 'n': a['n'] + b['n']}
    stat = dict()
    for szname, _ in size_ranges:
        stat[szname] = {'__all__': recall_empty()}
        for prop in properties:
            stat[szname][prop] = recall_empty()
            stat[szname]['~{}'.format(prop)] = recall_empty()
    chars = defaultdict(recall_empty)
    gts = ground_truth.splitlines()
    prs = prediction.splitlines()
    if len(gts) != len(prs):
        return error('number of lines not match')
    for i, (gt, pr) in enumerate(zip(gts, prs)):
        gt = json.loads(gt)
        try:
            pr = json.loads(pr)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(pr, dict):
            return error('line {} is not json object'.format(i + 1))
        if 'predictions' not in pr:
            return error('line {} does not contains key `predictions`'.format(i + 1))
        gt = gt['ground_truth']
        pr = pr['predictions']
        if not isinstance(pr, list):
            return error('line {} predictions is not a list'.format(i + 1))
        if len(pr) != len(gt):
            return error('line {} wrong predictions length'.format(i + 1))
        for j, (cgt, cpr) in enumerate(zip(gt, pr)):
            if not isinstance(cpr, list):
                return error('line {} prediction {} is not a list'.format(i + 1, j + 1))
            if recall_n[-1] < len(cpr):
                return error('line {} prediction {} contains too few candidates'.format(i + 1, j + 1))
            for k, s in enumerate(cpr):
                if not isinstance(s, six.text_type):
                    return error('line {} prediction {} item {} is not a text'.format(i + 1, j + 1, k + 1))
            thisrc = {'recalls': {n: 1 if cgt['text'] in cpr[:n] else 0 for n in recall_n}, 'n': 1}
            longsize = max(cgt['size'])
            for szname, range in size_ranges:
                if range[0] <= longsize and longsize < range[1]:
                    for prop in properties:
                        if prop not in cgt['properties']:
                            prop = '~{}'.format(prop)
                        stat[szname][prop] = recall_add(stat[szname][prop], thisrc)
                    stat[szname]['__all__'] = recall_add(stat[szname]['__all__'], thisrc)
            chars[cgt['text']] = recall_add(chars[cgt['text']], thisrc)
    return {'error': 0, 'statistics': stat, 'group_by_characters': chars}


def iou(bbox_0, bbox_1): # bbox is represented as (x, y, w, h)
    assert bbox_0[2] >= 0 and bbox_0[3] >= 0 and bbox_1[2] >= 0 and bbox_1[3] >= 0
    A0 = bbox_0[2] * bbox_0[3]
    A1 = bbox_1[2] * bbox_1[3]
    if A0 == 0 or A1 == 0:
        return 0
    Nw = min(bbox_0[0] + bbox_0[2], bbox_1[0] + bbox_1[2]) - max(bbox_0[0], bbox_1[0])
    Nh = min(bbox_0[1] + bbox_0[3], bbox_1[1] + bbox_1[3]) - max(bbox_0[1], bbox_1[1])
    AN = max(0, Nw) * max(0, Nh)
    return AN / (A0 + A1 - AN)
