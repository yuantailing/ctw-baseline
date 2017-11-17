# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import bisect
import codecs
import json
import math
import numpy as np
import os
import settings

from collections import defaultdict
from filename_mapper import mapper
from pythonapi import anno_tools, common_tools, eval_tools


def get_allowed_filename():
    md52old = dict()
    allowed_md5 = set()
    for folder in ['sign', 'nosign']:
        with open(os.path.join('md5sum_match', 'use_md5sum_{}.txt'.format(folder))) as f:
            for line in f:
                md5, filename = line.split()
                pre = ''
                if md5 in md52old:
                    pre = md52old[md5][0] + ','
                md52old[md5] = (pre + folder, filename[:-4])
    for folder in ['train', 'test', 'other', 'nosign']:
        counts = defaultdict(int)
        with open(os.path.join('md5sum_match', 'traffic_md5sum_{}.txt'.format(folder))) as f:
            for line in f:
                md5, filename = line.split()
                assert md5 in md52old
                origin_folder, filename = md52old[md5]
                counts[filename[:4]] += 1
                allowed_md5.add(md5)
        # print(folder, dict(counts))
    return set(md52old[md5][1] for md5 in allowed_md5)


def bbox2poly(bbox):
    x, y, w, h = bbox
    return [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]


def is_legal_char(imshape, logo_bbox):
    def g(char):
        x, y, w, h = bbox = char['adjusted_bbox']
        if logo_bbox and .5 < eval_tools.a_in_b(bbox, logo_bbox):
            return False
        assert w > 0 and h > 0
        if w <= 0 or h <= 0:
            return False
        if x >= imshape[1] or x + w <= 0 or y >= imshape[0] or y + h <= 0:
            return False
        if char['is_chinese']:
            assert 1 == len(char['text'])
            if 1 != len(char['text']):
                return False
        return True
    return g


def main():
    allowed_filename = get_allowed_filename()
    basename2cnt = dict()
    imgid2anno = dict()
    with codecs.open(settings.OVERALL_ANNOTATIONS, 'r', 'utf-8') as f:
        for line in f:
            image_anno = json.loads(line.strip())
            image_id = image_anno['image_id']
            if image_id[:3] == '204':
                logo_bbox = [810., 1800., 460., 100.]
                image_anno['ignore'].append({
                    'bbox': logo_bbox,
                    'polygon': bbox2poly(logo_bbox),
                })
            else:
                logo_bbox = None
            for i, blk in enumerate(image_anno['annotations']):
                image_anno['annotations'][i] = list(filter(is_legal_char((image_anno['height'], image_anno['width'], 3), logo_bbox), blk))
            image_anno['annotations'] = list(filter(lambda a: a, image_anno['annotations']))
            direction = image_id[0]
            onew = image_id[1:]
            oold = mapper.new2old[onew]
            complete_basename = '{}.{}'.format(oold, direction)
            if image_id[1:3] != '04' and complete_basename not in allowed_filename:
                continue
            is_chinese = 0
            is_not_chinese = 0
            for char in anno_tools.each_char(image_anno):
                if char['is_chinese']:
                    is_chinese += 1
                else:
                    is_not_chinese += 1
            if 0 == is_chinese:
                continue
            basename2cnt[complete_basename] = [is_chinese, is_not_chinese]
            imgid2anno[image_id] = image_anno
    streets = []
    current_street = []

    def feature(basename):
        assert 25 == len(basename)
        cityid = basename[:4]
        streetid = basename[4:8]
        dateid = basename[8:14]
        timeid = basename[14:23]
        direction = basename[-1]
        return (cityid, streetid, dateid, timeid, direction)

    def like(n1, n2):
        f1, f2 = feature(n1), feature(n2)
        return (f1[0], f1[1], f1[2]) == (f2[0], f2[1], f2[2])
    for i, basename in enumerate(sorted(basename2cnt.keys())):
        if i == 0 or like(last_basename, basename):
            current_street.append(basename)
        else:
            streets.append(current_street)
            current_street = [basename]
        last_basename = basename
    streets.append(current_street)

    def count_chinese(a):
        return sum([basename2cnt[basename][0] for basename in a])

    np.random.seed(0)
    train = []
    tests = [[], [], []]
    requires = [101000, 101000, 50500]

    basenames = sorted(mapper.old2new.keys())
    for i, st in enumerate(streets):
        ntest = 100
        nval = 50
        ntrainval = 1200
        ninterval = 5
        if ntest * 2 + ninterval + ntrainval < len(st):
            while 0 < len(st):
                j = np.random.randint(0, 2)
                tests[j] += st[0:ntest]
                tests[1 - j] += st[ntest:ntest * 2]
                lastname = st[ntest * 2 - 1].split('.')[0]
                lastpos = bisect.bisect(basenames, lastname) - 1
                assert basenames[lastpos] == lastname
                st = st[ntest * 2:]
                j = 0
                while True:
                    name = st[j].split('.')[0]
                    pos = bisect.bisect(basenames, name) - 1
                    assert basenames[pos] == name
                    if pos - lastpos <= ninterval:
                        j += 1
                    else:
                        break
                st = st[j:]
                hi = ntrainval
                if len(st) < ntrainval * (math.sqrt(5) + 1) / 2 + ntest * 2 + ninterval * 2:
                    hi = len(st)
                j = np.random.randint(0, 2)
                train += st[j * nval:hi - nval + j * nval]
                tests[2] += st[0:j * nval] + st[hi - nval + j * nval:hi]
                lastname = st[hi-1].split('.')[0]
                st = st[hi:]
                if 0 < len(st):
                    lastpos = bisect.bisect(basenames, lastname) - 1
                    j = 0
                    while True:
                        name = st[j].split('.')[0]
                        pos = bisect.bisect(basenames, name) - 1
                        assert(basenames[pos] == name)
                        if pos - lastpos <= ninterval:
                            j += 1
                        else:
                            break
                    st = st[j:]
            streets[i] = []
    counts = [count_chinese(test) for test in tests]

    while (np.array(counts) < np.array(requires)).any():
        i = np.random.randint(0, len(streets))
        j = np.random.randint(0, len(requires))
        if counts[j] >= requires[j]:
            continue
        st = streets[i]
        if 300 < len(st):
            continue
        tests[j] += streets.pop(i)
        counts[j] = count_chinese(tests[j])
    for st in streets:
        train += st

    print('train', len(train), count_chinese(train))
    print('val', len(tests[2]), count_chinese(tests[2]))
    print('test_1', len(tests[0]), count_chinese(tests[0]))
    print('test_2', len(tests[1]), count_chinese(tests[1]))

    def basename2imgid(bn):
        a, direction = bn.split('.')
        return '{}{}'.format(direction, mapper.old2new[a])
    train = sorted(map(basename2imgid, train))
    val = sorted(map(basename2imgid, tests[2]))
    test_1 = sorted(map(basename2imgid, tests[0]))
    test_2 = sorted(map(basename2imgid, tests[1]))

    def toinfo(anno):
        keys = list(anno.keys())
        keys.remove('annotations')
        keys.remove('ignore')
        return {k: anno[k] for k in keys}

    with open(settings.DATA_LIST, 'w') as f:
        def g(ds):
            return [toinfo(imgid2anno[imgid]) for imgid in ds]
        r = {'train': g(train), 'val': g(val), 'test_cls': g(test_1), 'test_det': g(test_2)}
        json.dump(r, f, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)

    with open(settings.TRAIN, 'w') as f:
        for imgid in train:
            f.write(common_tools.to_jsonl(imgid2anno[imgid]))
            f.write('\n')
    with open(settings.VAL, 'w') as f:
        for imgid in val:
            f.write(common_tools.to_jsonl(imgid2anno[imgid]))
            f.write('\n')
    with open(settings.TEST_CLASSIFICATION, 'w') as f, open(settings.TEST_CLASSIFICATION_GT, 'w') as fgt:
        for imgid in test_1:
            anno = imgid2anno[imgid]
            proposals = []
            gt = []
            for char in anno_tools.each_char(anno):
                if not char['is_chinese']:
                    continue
                proposals.append({'adjusted_bbox': char['adjusted_bbox'], 'polygon': char['polygon']})
                gt.append({'text': char['text'], 'attributes': char['attributes'], 'size': char['adjusted_bbox'][-2:]})
            anno.pop('annotations')
            anno.pop('ignore')
            anno['proposals'] = proposals
            f.write(common_tools.to_jsonl(anno))
            f.write('\n')
            anno.pop('proposals')
            anno['ground_truth'] = gt
            fgt.write(common_tools.to_jsonl(anno))
            fgt.write('\n')
    with open(settings.TEST_DETECTION_GT, 'w') as f:
        for imgid in test_2:
            f.write(common_tools.to_jsonl(imgid2anno[imgid]))
            f.write('\n')


if __name__ == '__main__':
    main()
