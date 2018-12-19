# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
import operator
import six

from . import anno_tools, common_tools
from collections import defaultdict
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def classification_recall(ground_truth, prediction, recall_n, attributes, size_ranges):
    def error(s):
        return {'error': 1, 'msg': s}

    def recall_empty():
        return {'recalls': {n: 0 for n in recall_n}, 'n': 0}

    def recall_add(a, b):
        return {'recalls': {n: a['recalls'][n] + b['recalls'][n] for n in recall_n}, 'n': a['n'] + b['n']}

    stat = dict()
    for szname, _ in size_ranges:
        stat[szname] = {
            'attributes': [recall_empty() for _ in range(2 ** len(attributes))],
            'texts': defaultdict(recall_empty),
        }
    gts = ground_truth.splitlines()
    prs = prediction.splitlines()
    if len(gts) != len(prs):
        return error('number of lines not match, gt={} vs. pr={}'.format(len(gts), len(prs)))
    for i, (gt, pr) in enumerate(zip(gts, prs)):
        gt = json.loads(gt)
        try:
            pr = json.loads(pr)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(pr, dict):
            return error('line {} is not json object'.format(i + 1))
        if 'predictions' not in pr:
            return error('line {} does not contain key `predictions`'.format(i + 1))
        gt = gt['ground_truth']
        pr = pr['predictions']
        if not isinstance(pr, list):
            return error('line {} predictions is not an array'.format(i + 1))
        if len(pr) != len(gt):
            return error('line {} number of predictions not match, gt={} vs. dt={}'.format(i + 1, len(gt), len(pr)))
        for j, (cgt, cpr) in enumerate(zip(gt, pr)):
            if not isinstance(cpr, list):
                return error('line {} prediction {} is not an array'.format(i + 1, j + 1))
            if recall_n[-1] < len(cpr):
                return error('line {} prediction {} contains too few candidates'.format(i + 1, j + 1))
            for k, s in enumerate(cpr):
                if not isinstance(s, six.text_type):
                    return error('line {} prediction {} item {} is not a text'.format(i + 1, j + 1, k + 1))
            thisrc = {'recalls': {n: 1 if cgt['text'] in cpr[:n] else 0 for n in recall_n}, 'n': 1}
            longsize = max(cgt['size'])
            for szname, rng in size_ranges:
                if rng[0] <= longsize < rng[1]:
                    k = 0
                    for attr in cgt['attributes']:
                        k += 2 ** attributes.index(attr)
                    stat[szname]['attributes'][k] = recall_add(stat[szname]['attributes'][k], thisrc)
                stat[szname]['texts'][cgt['text']] = recall_add(stat[szname]['texts'][cgt['text']], thisrc)
    for szname, _ in size_ranges:
        stat[szname]['texts'] = dict(stat[szname]['texts'])
    return {'error': 0, 'performance': stat}


def iou(bbox_0, bbox_1):  # bbox is represented as (x, y, w, h)
    assert bbox_0[2] >= 0 and bbox_0[3] >= 0 and bbox_1[2] >= 0 and bbox_1[3] >= 0
    A0 = bbox_0[2] * bbox_0[3]
    A1 = bbox_1[2] * bbox_1[3]
    if 0 == A0 or 0 == A1:
        return 0
    Nw = min(bbox_0[0] + bbox_0[2], bbox_1[0] + bbox_1[2]) - max(bbox_0[0], bbox_1[0])
    Nh = min(bbox_0[1] + bbox_0[3], bbox_1[1] + bbox_1[3]) - max(bbox_0[1], bbox_1[1])
    AN = max(0, Nw) * max(0, Nh)
    return AN / (A0 + A1 - AN)


def a_in_b(bbox_0, bbox_1):
    assert bbox_0[2] >= 0 and bbox_0[3] >= 0 and bbox_1[2] >= 0 and bbox_1[3] >= 0
    A0 = bbox_0[2] * bbox_0[3]
    if 0 == A0:
        return 0
    Nw = min(bbox_0[0] + bbox_0[2], bbox_1[0] + bbox_1[2]) - max(bbox_0[0], bbox_1[0])
    Nh = min(bbox_0[1] + bbox_0[3], bbox_1[1] + bbox_1[3]) - max(bbox_0[1], bbox_1[1])
    AN = max(0, Nw) * max(0, Nh)
    return AN / A0


def detection_mAP(ground_truth, detection, attributes, size_ranges, max_det, iou_thresh, proposal=False, echo=False):
    def error(s):
        return {'error': 1, 'msg': s}

    def AP_empty():
        return {'n': 0, 'dt': [], 'attributes': [{'n': 0, 'recall': 0} for _ in range(2 ** len(attributes))]}

    def AP_compute(m):
        if 0 == m['n']:
            return None, []
        acc = []
        rc_inc = []
        m['dt'].sort(key=lambda t: (-t[2], t[1], t[0]))  # sort order by matched ASC, intended to prevent from cheating
        match_cnt = 0
        for i, (matched, _, _) in enumerate(m['dt']):
            assert matched in (0, 1)
            match_cnt += matched
            acc.append(match_cnt / (i + 1))
            rc_inc.append(matched)
        for i in range(len(acc) - 1, 0, -1):
            acc[i - 1] = max(acc[i - 1], acc[i])
        AP = 0
        curve = []
        for a, r in zip(acc, rc_inc):
            AP += a * r
            if 1 == r:
                curve.append(a)
        return AP / m['n'], curve

    eval_type = 'proposals' if proposal else 'detections'
    m = dict()
    for szname, _ in size_ranges:
        m[szname] = defaultdict(AP_empty)
    AP_imgs = {szname: dict() for szname, _ in size_ranges}

    gts = ground_truth.splitlines()
    dts = detection.splitlines()
    if len(gts) != len(dts):
        return error('number of lines not match: {} expected, {} loaded'.format(len(gts), len(dts)))

    for i, (gt, dt) in enumerate(zip(gts, dts)):
        if echo and 0 == i % 200:
            print(i, '/', len(gts))

        try:
            dt = json.loads(dt)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(dt, dict):
            return error('line {} is not json object'.format(i + 1))
        if eval_type not in dt:
            return error('line {} does not contain key `{}`'.format(i + 1, eval_type))
        dt = dt[eval_type]
        if not isinstance(dt, list):
            return error('line {} {} is not an array'.format(i + 1, eval_type))
        if len(dt) > max_det:
            return error('line {} number of {} exceeds limit ({})'.format(i + 1, eval_type, max_det))
        for j, char in enumerate(dt):
            if not isinstance(char, dict):
                return error('line {} {} candidate {} is not an object'.format(i + 1, eval_type, j + 1))
            if not proposal and 'text' not in char:
                return error('line {} {} candidate {} does not contain key `text`'.format(i + 1, eval_type, j + 1))
            if 'score' not in char:
                return error('line {} {} candidate {} does not contain key `score`'.format(i + 1, eval_type, j + 1))
            if 'bbox' not in char:
                return error('line {} {} candidate {} does not contain key `bbox`'.format(i + 1, eval_type, j + 1))
            if not isinstance(char['text'], six.text_type):
                return error('line {} {} candidate {} text is not text-type'.format(i + 1, eval_type, j + 1))
            if not isinstance(char['score'], (int, float)):
                return error('line {} {} candidate {} score is neither int nor float'.format(i + 1, eval_type, j + 1))
            if not isinstance(char['bbox'], list):
                return error('line {} {} candidate {} bbox is not an array'.format(i + 1, eval_type, j + 1))
            if 4 != len(char['bbox']):
                return error('line {} {} candidate {} bbox is illegal'.format(i + 1, eval_type, j + 1))
            for t in char['bbox']:
                if not isinstance(t, (int, float)):
                    return error('line {} {} candidate {} bbox is illegal'.format(i + 1, eval_type, j + 1))
            if char['bbox'][2] <= 0 or char['bbox'][3] <= 0:
                return error('line {} {} candidate {} bbox w or h <= 0'.format(i + 1, eval_type, j + 1))

        dt.sort(key=operator.itemgetter('score'), reverse=True)  # sort must be stable, otherwise mAP will be slightly different
        dt = [(o['bbox'], o.get('text'), o['score']) for o in dt]

        gtobj = json.loads(gt)
        ig = [o['bbox'] for o in gtobj['ignore']]
        gt = []
        for char in anno_tools.each_char(gtobj):
            if char['is_chinese']:
                gt.append((char['adjusted_bbox'], char['text'], char['attributes']))

        dt_matches = [[] for i in range(len(dt))]
        dt_ig = [False] * len(dt)
        for i_dt, dtchar in enumerate(dt):
            for i_gt, gtchar in enumerate(gt):
                if proposal or dtchar[1] == gtchar[1]:
                    miou = iou(dtchar[0], gtchar[0])
                    if miou > iou_thresh:
                        dt_matches[i_dt].append((-miou, i_gt))
            for igchar in ig:
                miou = a_in_b(dtchar[0], igchar)
                if miou > iou_thresh:
                    dt_ig[i_dt] = True
        for matches in dt_matches:
            matches.sort()

        for szname, size_range in size_ranges:
            def in_size(bbox):
                longsize = max(bbox[2], bbox[3])
                return size_range[0] <= longsize < size_range[1]

            dt_matched = [0 if in_size(o[0]) and False == b else 2 for o, b in zip(dt, dt_ig)]
            gt_taken = [(0, None) if in_size(o[0]) else (2, None) for o in gt]
            for i_dt, matches in enumerate(dt_matches):
                for _, i_gt in matches:
                    if 1 != dt_matched[i_dt] and 1 != gt_taken[i_gt][0]:
                        if 0 == gt_taken[i_gt][0]:
                            dt_matched[i_dt] = 1
                            gt_taken[i_gt] = (1, i_dt)
                        else:
                            dt_matched[i_dt] = 2
            m_img = {'n': 0, 'dt': []}
            for i_dt, (dtchar, match_status) in enumerate(zip(dt, dt_matched)):
                if 2 != match_status:
                    m[szname][dtchar[1]]['dt'].append((match_status, i_dt, dtchar[2]))
                    m_img['dt'].append((match_status, i_dt, dtchar[2]))
            top_dt = [i for i, ms in enumerate(dt_matched) if 2 != ms]
            top_dt = set(top_dt[:sum([2 != taken for taken, _ in gt_taken])])
            for gtchar, (taken, dt_id) in zip(gt, gt_taken):
                if 2 != taken:
                    thism = m[szname][gtchar[1]]
                    thism['n'] += 1
                    m_img['n'] += 1
                    k = 0
                    for attr in gtchar[2]:
                        k += 2 ** attributes.index(attr)
                    thism['attributes'][k]['n'] += 1
                    if 0 != taken and dt_id in top_dt:
                        thism['attributes'][k]['recall'] += 1
            AP_img, _ = AP_compute(m_img)
            AP_imgs[szname][gtobj['image_id']] = AP_img

    performance = dict()
    for szname, _ in size_ranges:
        n = 0
        mAP = 0.
        szattrs = AP_empty()['attributes']
        texts = dict()
        stat_all = AP_empty()
        for text, stat in m[szname].items():
            if 0 == stat['n']:
                continue
            n += stat['n']
            AP, _ = AP_compute(stat)
            mAP += AP * stat['n']
            for k, o in enumerate(szattrs):
                o['n'] += stat['attributes'][k]['n']
                o['recall'] += stat['attributes'][k]['recall']
            texts[text] = {'AP': AP, 'n': stat['n']}
            stat_all['n'] += stat['n']
            stat_all['dt'] += stat['dt']
        AP_all, AP_curve = AP_compute(stat_all)
        a_micro = list(filter(lambda x: x is not None, AP_imgs[szname].values()))
        mAP_micro = None if 0 == len(a_micro) else sum(a_micro) / len(a_micro)
        performance[szname] = {
            'n': n,
            'mAP': AP_all if proposal else mAP / n if 0 != n else None,
            'attributes': szattrs,
            'texts': texts,
            'AP': AP_all,
            'mAP_micro': mAP_micro,
            'AP_curve': AP_curve,
        }
    return {'error': 0, 'performance': performance}


def iou_polygon(poly_a, poly_b):
    AN = (poly_a & poly_b).area
    if AN == 0:
        return 0
    A0 = poly_a.area
    A1 = poly_b.area
    return max(0., min(1., AN / (A0 + A1 - AN)))


def a_in_b_polygon(poly_a, poly_b):
    AN = (poly_a & poly_b).area
    if AN == 0:
        return 0
    A0 = poly_a.area
    if A0 <= 0:
        return 0
    return max(0., min(1., AN / A0))


def textline_AED(ground_truth, detection, max_det, max_vertices, iou_thresh, echo=False):
    def error(s):
        return {'error': 1, 'msg': s}

    def edit_distance(s1, s2):
        m, n = len(s1) + 1, len(s2) + 1
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            dp[0][i] = i
        for i in range(m):
            dp[i][0] = i
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1))
        return dp[m - 1][n - 1]

    def bbox2polygon(bbox):
        points = []
        points.append((bbox[0], bbox[1]))
        points.append((bbox[0], bbox[1] + bbox[3]))
        points.append((bbox[0] + bbox[2], bbox[1] + bbox[3]))
        points.append((bbox[0] + bbox[2], bbox[1]))
        return points

    def gt_helper(line):
        line = [char for char in line if char['is_chinese']]
        if not line:
            return None
        points = np.array([p for char in line for p in char['polygon']])
        hull = ConvexHull(points)
        polygon = [points[i].tolist() for i in hull.vertices]
        text = common_tools.reduce_sum(char['text'] for char in line)
        return Polygon(polygon), text

    eval_type = 'textlines'

    gts = ground_truth.splitlines()
    dts = detection.splitlines()
    if len(gts) != len(dts):
        return error('number of lines not match: {} expected, {} loaded'.format(len(gts), len(dts)))

    sum_ed = 0
    sum_gt = 0
    for i, (gt, dt) in enumerate(zip(gts, dts)):
        if echo and 0 == i % 200:
            print(i, '/', len(gts))

        try:
            dt = json.loads(dt)
        except:
            return error('line {} is not legal json'.format(i + 1))
        if not isinstance(dt, dict):
            return error('line {} is not json object'.format(i + 1))
        if eval_type not in dt:
            return error('line {} does not contain key `{}`'.format(i + 1, eval_type))
        dt = dt[eval_type]
        if not isinstance(dt, list):
            return error('line {} {} is not an array'.format(i + 1, eval_type))
        if len(dt) > max_det:
            return error('line {} number of {} exceeds limit ({})'.format(i + 1, eval_type, max_det))
        for j, line in enumerate(dt):
            if not isinstance(line, dict):
                return error('line {} {} candidate {} is not an object'.format(i + 1, eval_type, j + 1))
            if 'text' not in line:
                return error('line {} {} candidate {} does not contain key `text`'.format(i + 1, eval_type, j + 1))
            if 'polygon' not in line:
                return error('line {} {} candidate {} does not contain key `polygon`'.format(i + 1, eval_type, j + 1))
            if not isinstance(line['text'], six.text_type):
                return error('line {} {} candidate {} text is not text-type'.format(i + 1, eval_type, j + 1))
            if not isinstance(line['polygon'], list):
                return error('line {} {} candidate {} polygon is not an array'.format(i + 1, eval_type, j + 1))
            if len(line['polygon']) < 3:
                return error('line {} {} polygon {} contains too less vertices'.format(i + 1, eval_type, j + 1))
            if max_vertices < len(line['polygon']):
                return error('line {} {} polygon {} contains too many vertices'.format(i + 1, eval_type, j + 1))
            for t in line['polygon']:
                if not isinstance(t, list) or 2 != len(t):
                    return error('line {} {} polygon {} is ill formatted'.format(i + 1, eval_type, j + 1))
                for x in t:
                    if not isinstance(x, (int, float)):
                        return error('line {} {} polygon {} is ill formatted'.format(i + 1, eval_type, j + 1))
            polygon = Polygon(line['polygon'])
            if not polygon.is_valid:
                return error('line {} {} polygon {} is not valid (e.g., has self-intersection)'.format(i + 1, eval_type, j + 1))

        gtobj = json.loads(gt)
        dt = [(Polygon(line['polygon']), line['text']) for line in dt]
        gt = [gt_helper(line) for line in gtobj['annotations']]
        gt = [o for o in gt if o is not None]
        ig = [Polygon(bbox2polygon(o['bbox'])) for o in gtobj['ignore']]

        dt_matches = [[] for i in range(len(dt))]
        dt_ig = [False] * len(dt)
        for i_dt, dtline in enumerate(dt):
            for i_gt, gtline in enumerate(gt):
                miou = iou_polygon(dtline[0], gtline[0])
                if miou > iou_thresh:
                    dt_matches[i_dt].append((-miou, i_gt))
            for igline in ig:
                miou = a_in_b_polygon(dtline[0], igline)
                if miou > iou_thresh:
                    dt_ig[i_dt] = True
        for matches in dt_matches:
            matches.sort()

        dt_matched = [0 if False == b else 2 for b in dt_ig]
        gt_taken = [(0, None) for o in gt]
        for i_dt, matches in enumerate(dt_matches):
            for _, i_gt in matches:
                if 1 != dt_matched[i_dt] and 1 != gt_taken[i_gt][0]:
                    dt_matched[i_dt] = 1
                    gt_taken[i_gt] = (1, i_dt)
        len_gt = 0
        len_fn = 0    # gt not matched
        len_tp_ed = 0 # matched edit distance
        len_fp = 0    # dt not matched
        len_ig = 0    # dt ignored
        for i_gt, (gt_status, i_dt) in enumerate(gt_taken):
            len_gt += len(gt[i_gt][1])
            if gt_status == 0:
                len_fn += len(gt[i_gt][1])
            else:
                len_tp_ed += edit_distance(gt[i_gt][1], dt[i_dt][1])
        for i_dt, dt_status in enumerate(dt_matched):
            if dt_status == 0:
                len_fp += len(dt[i_dt][1])
            elif dt_status == 2:
                len_ig += len(dt[i_dt][1])
        ed = len_tp_ed + len_fp + len_fn
        sum_ed += ed
        sum_gt += len_gt
    return {
        'AED': sum_ed / len(gts),
        'instance_per_image': sum_gt / len(gts),
    }
