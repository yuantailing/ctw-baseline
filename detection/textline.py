# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import json
import numpy as np
import os
import settings

from pythonapi import common_tools, eval_tools
from scipy.spatial import ConvexHull


def line_merge(detections):
    def match(a, b, row=False):
        abox = a['bbox']
        bbox = b['bbox']
        if not (.667 < abox[2] * abox[3] / (bbox[2] * bbox[3]) < 1.5):
            return False
        amid = (abox[0] + abox[2] / 2, abox[1] + abox[3] / 2)
        bmid = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        line = (bmid[0] - amid[0], bmid[1] - amid[1])
        if row:
            if abs(line[0]) > abs(line[1]) * .4:
                return False
        else:
            if abs(line[0]) * abox[3] < abs(line[1]) * abox[2] * 4:
                return False

        if row:
            allow = (abox[0] - abox[2] * .2, abox[1] - abox[3] * 2., abox[2] * 1.4, abox[3] * 5.)
        else:
            allow = (abox[0] - abox[2] * 2., abox[1] - abox[3] * .2, abox[2] * 5., abox[3] * 1.4)
        if .5 < eval_tools.a_in_b(bbox, allow):
            return True
        return False

    all = detections
    a2g = list(range(len(all)))
    for i, a in enumerate(all):
        for j, b in enumerate(all):
            if match(a, b) or match(b, a):
                x = a2g[j]
                for k, gid in enumerate(a2g):
                    if gid == x:
                        a2g[k] = a2g[i]
    linelength = []
    for i, gid in enumerate(a2g):
        linelength.append(len([0 for id in a2g if id == gid]))
    a2row = list(range(len(all)))
    for i, a in enumerate(all):
        for j, b in enumerate(all):
            if match(a, b, True) or match(b, a, True):
                x = a2row[j]
                for k, rid in enumerate(a2row):
                    if rid == x:
                        a2row[k] = a2row[i]

    rows = []
    counted = set()
    for i, a in enumerate(all):
        rid = a2row[i]
        if rid in counted:
            continue
        counted.add(rid)
        row = []
        sumlinelen = 0
        cntlinelen = 0
        for j, b in enumerate(all):
            if a2row[j] == rid:
                row.append(b)
                if b['text']:
                    sumlinelen += linelength[j]
                    cntlinelen += 1
        if cntlinelen < 2 or sumlinelen > cntlinelen * 2:
            continue
        for j, b in enumerate(all):
            if a2row[j] == rid:
                a2g[j] = -1
        row.sort(key=lambda o: o['bbox'][1])
        rows.append(row)
    rows.sort(key=lambda row: row[0]['bbox'][0])

    lines = []
    counted = {-1}
    for i, a in enumerate(all):
        gid = a2g[i]
        if gid in counted:
            continue
        counted.add(gid)
        line = []
        for j, b in enumerate(all):
            if a2g[j] == gid:
                line.append(b)
        line.sort(key=lambda o: o['bbox'][0])
        if len(line):
            lines.append(line)
    lines.sort(key=lambda line: line[0]['bbox'][1])

    res = []
    for line in lines + rows:
        line = [char for char in line if char['score'] >= .2]
        text = ''
        sum_score = 0.
        cnt = 0
        for char in line:
            text += char['text']
            sum_score += char['score']
            cnt += 1
        if not text:
            continue
        if sum_score / cnt < .5:
            continue
        points = []
        for char in line:
            bbox = char['bbox']
            points.append((bbox[0], bbox[1]))
            points.append((bbox[0], bbox[1] + bbox[3]))
            points.append((bbox[0] + bbox[2], bbox[1] + bbox[3]))
            points.append((bbox[0] + bbox[2], bbox[1]))
        points = np.array(points, dtype=np.float32)
        hull = ConvexHull(points)
        polygon = [points[i].tolist() for i in hull.vertices]
        if len(polygon) > settings.TEXTLINE_MAX_VERTICES:
            rect = cv2.minAreaRect(points)
            polygon = cv2.boxPoints(rect).tolist()
        res.append({'polygon': polygon, 'text': text, 'avg_score': sum_score / cnt})
    res.sort(key=lambda o: (-len(o['text']), -o['avg_score']))
    return res[:settings.TEXTLINE_MAX_DET]


if __name__ == '__main__':
    LINE_PLACE_THRESHOLD = .2
    NMS = .5

    print('load detection results')
    all = []
    with open(os.path.join(settings.PRODUCTS_ROOT, 'detections.jsonl')) as f:
        for i_line, line in enumerate(f):
            jdata = json.loads(line)
            dt, pr = jdata['detections'], jdata['proposals']
            a = [o for o in dt + pr if o['score'] >= LINE_PLACE_THRESHOLD]
            a.sort(key=lambda o: -o['score'])
            na = []
            for o in a:
                covered = 0
                for no in na:
                    covered += eval_tools.a_in_b(o['bbox'], no['bbox'])
                    if covered > NMS:
                        break
                if covered <= NMS:
                    na.append(o)
            all.append(na)

    print('merge textlines')
    with open(os.path.join(settings.PRODUCTS_ROOT, 'textlines.jsonl'), 'w') as f:
        for detections in all:
            lines = line_merge(detections)
            f.write(common_tools.to_jsonl({'textlines': lines}))
            f.write('\n')
