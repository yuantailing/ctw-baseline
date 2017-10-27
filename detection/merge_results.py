# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import darknet_tools
import functools
import json
import os
import settings
import six
import subprocess
import sys

from collections import defaultdict
from pythonapi import common_tools, eval_tools
from six.moves import cPickle


caches = {
    'UNMERGED': os.path.join(settings.PRODUCTS_ROOT, 'unmerged.pkl'),
}


def clear_caches():
    for file_path in caches.values():
        if os.path.isfile(file_path):
            os.unlink(file_path)


def read():
    file_paths = []
    pkl_is_newer = True
    for split_id in range(settings.TEST_SPLIT_NUM):
        darknet_results_out = darknet_tools.append_before_ext(settings.DARKNET_RESULTS_OUT, '.{}'.format(split_id))
        result_file_path = os.path.join(settings.DARKNET_RESULTS_DIR, '{}.txt'.format(darknet_results_out))
        file_paths.append(result_file_path)
        if not common_tools.exists_and_newer(caches['UNMERGED'], result_file_path):
            pkl_is_newer = False
    if pkl_is_newer:
        with open(caches['UNMERGED'], 'rb') as f:
            return cPickle.load(f)
    all = defaultdict(lambda: [])
    imshape = (2048, 2048, 3)
    removal = (1., 3.)
    size_ranges = ((8., 128.), (28., float('inf')))
    levelmap = dict()
    for level_id, (cropratio, cropoverlap) in enumerate(settings.TEST_CROP_LEVELS):
        cropshape = (settings.TEST_IMAGE_SIZE // cropratio, settings.TEST_IMAGE_SIZE // cropratio)
        for o in darknet_tools.get_crop_bboxes(imshape, cropshape, (cropoverlap, cropoverlap)):
            levelmap[level_id, o['name']] = (o['xlo'], o['ylo'], cropshape[1], cropshape[0])

    def bounded_bbox(bbox):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(imshape[1], x1), min(imshape[0], y1)
        return (x0, y0, x1 - x0, y1 - y0)

    def read_one(result_file_path):
        with open(result_file_path) as f:
            lines = f.read().splitlines()
        one = []
        for line in lines:
            file_path, cate_id, x, y, w, h, prob = line.split()
            image_id, level_id, crop_name = os.path.splitext(os.path.basename(file_path))[0].split('_', maxsplit=2)
            level_id = int(level_id)
            cx, cy, cw, ch = levelmap[level_id, crop_name]
            cate_id = int(cate_id)
            x, y, w, h, prob = float(x), float(y), float(w), float(h), float(prob)
            longsize = max(x, y)
            size_range = size_ranges[level_id]
            if longsize < size_range[0] or size_range[1] <= longsize:
                continue
            rm = removal[level_id]
            if (cx != 0 and x < rm) or (cy != 0 and y < rm) or (cx + cw != imshape[1] and x + w + rm >= cw) or (cy + ch != imshape[0] and y + h + rm >= ch):
                continue
            real_bbox = bounded_bbox((x + cx, y + cy, w, h))
            if real_bbox[2] > 0 and real_bbox[3] > 0:
                all[image_id].append({'image_id': image_id, 'cate_id': cate_id, 'prob': prob, 'bbox': real_bbox})

    for file_path in file_paths:
        read_one(file_path)
    all = dict(all)
    with open(caches['UNMERGED'], 'wb') as f:
        cPickle.dump(all, f)
    return all


def do_nms_sort(unmerged, nms):
    all = dict()
    i_time = 0
    for image_id, proposals in unmerged.items():
        if i_time % 200 == 0:
            print('nms sort', i_time, '/', len(unmerged))
        i_time += 1
        cates = defaultdict(lambda: [])
        for proposal in proposals:
            cates[proposal['cate_id']].append(proposal)
        for cate_id, proposal in cates.items():
            a = sorted(proposal, key=lambda o: -o['prob'])
            ignored = set()
            for i in range(len(a)):
                if i in ignored:
                    continue
                for j in range(i + 1, len(a)):
                    if j not in ignored and eval_tools.iou(a[i]['bbox'], a[j]['bbox']) > nms:
                        ignored.add(j)
            cates[cate_id] = [o for i, o in enumerate(a) if i not in ignored]
        all[image_id] = functools.reduce(lambda a, b: a + b, cates.values(), [])
    return all


def write(nms_sorted):
    with open(settings.CATES) as f:
        cates = json.load(f)

    with open(os.path.join(settings.PRODUCTS_ROOT, 'detections.jsonl'), 'w') as f:
        for image_id in sorted(nms_sorted.keys()):
            detections = nms_sorted[image_id]
            detections.sort(key=lambda o: (-o['prob'], o['cate_id'], o['bbox']))
            f.write(common_tools.to_jsonl({
                image_id: {
                    'detections': [{
                        'text': cates[dt['cate_id']]['text'],
                        'bbox': dt['bbox'],
                        'score': dt['prob'],
                    } for dt in detections[:settings.MAX_DET_PER_IMAGE]],
                },
            }))
            f.write('\n')


def main():
    if not six.PY3:
        args = ['python3'] + sys.argv
        print(*args)
        p = subprocess.Popen(args, shell=False)
        p.wait()
        assert 0 == p.returncode
        return

    clear_caches()

    print('load darknet output')
    unmerged = read()

    print('do nms sort')
    nms_sorted = do_nms_sort(unmerged, .45)

    print('write results')
    write(nms_sorted)


if __name__ == '__main__':
    main()
