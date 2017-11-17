# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import json
import numpy as np
import os
import predictions2html
import random
import settings
import six

from collections import defaultdict
from pythonapi import anno_tools, common_tools
from six.moves import cPickle


def main():
    assert six.PY3
    random.seed(0)

    if not os.path.isfile(settings.DATASET_CROPPED):
        lines = []
        with open(settings.TRAIN) as f:
            lines += [('train', s) for s in f.read().splitlines()]
        with open(settings.VAL) as f:
            lines += [('val', s) for s in f.read().splitlines()]
        with open(settings.TEST_CLASSIFICATION) as f1, open(settings.TEST_CLASSIFICATION_GT) as f2:
            prs = f1.read().splitlines()
            gts = f2.read().splitlines()
            lines += [('test_cls', pr, gt) for pr, gt in zip(prs, gts)]
        with open(settings.TEST_DETECTION_GT) as f:
            lines += [('test_det', s) for s in f.read().splitlines()]
        all = [[] for _ in lines]
        def load_train(i):
            if i % 100 == 0:
                print('dataset', i, '/', len(lines))
            line = lines[i]
            if line[0] == 'test_cls':
                prs, gts = line[1:]
                prs, gts = json.loads(prs), json.loads(gts)
                image = cv2.imread(os.path.join(settings.TEST_IMAGE_DIR, prs['file_name']))
                assert image.shape == (prs['height'], prs['width'], 3)
                for pr, gt in zip(prs['proposals'], gts['ground_truth']):
                    cropped = predictions2html.crop(image, pr['adjusted_bbox'], 32)
                    all[i].append([cropped, gt['attributes']])
            else:
                anno = json.loads(line[1].strip())
                image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR if line[0] in {'train', 'val'} else settings.TEST_IMAGE_DIR, anno['file_name']))
                assert image.shape == (anno['height'], anno['width'], 3)
                for char in anno_tools.each_char(anno):
                    if not char['is_chinese']:
                        continue
                    cropped = predictions2html.crop(image, char['adjusted_bbox'], 32)
                    all[i].append([cropped, char['attributes']])
        common_tools.multithreaded(load_train, range(len(lines)), num_thread=8)
        all = common_tools.reduce_sum(all)
        with open(settings.DATASET_CROPPED, 'wb') as f:
            cPickle.dump(all, f)

    with open(settings.DATASET_CROPPED, 'rb') as f:
        all = cPickle.load(f)
    belongs = defaultdict(list)
    for i, (_, attrs) in enumerate(all):
        for attr in settings.ATTRIBUTES:
            if attr in attrs:
                belongs[attr].append(i)
            else:
                belongs['not-{}'.format(attr)].append(i)
    x, y, cropsize = 25, 20, 32
    for attr, imgset in sorted(belongs.items()):
        imgset = random.sample(imgset, x * y)
        if not os.path.isdir(settings.ATTR_SAMPLE_DIR):
            os.makedirs(settings.ATTR_SAMPLE_DIR)
        file_path = os.path.join(settings.ATTR_SAMPLE_DIR, '{}.png'.format(attr))
        print(file_path)
        canvas = np.zeros((y * cropsize, x * cropsize, 3), dtype=np.uint8)
        for i, j in enumerate(imgset):
            cropped = all[j][0]
            resized = cv2.resize(cropped, (cropsize, cropsize))
            canvas[(i // x) * cropsize:(i // x + 1) * cropsize, (i % x) * cropsize:(i % x + 1) * cropsize] = resized
        cv2.imwrite(file_path, canvas)


def crop_no_concat():
    assert six.PY3
    random.seed(0)

    belongs = defaultdict(list)
    with open(settings.TRAIN) as f:
        lines = f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()
    for line in lines:
        anno = json.loads(line.strip())
        i = 0
        for char in anno_tools.each_char(anno):
            if not char['is_chinese']:
                continue
            for attr in settings.ATTRIBUTES:
                if attr in char['attributes']:
                    belongs[attr].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
                else:
                    belongs['not-{}'.format(attr)].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
            i += 1

    for attr, imgset in sorted(belongs.items()):
        imgset = random.sample(imgset, 200)
        root = os.path.join(settings.ATTR_SAMPLE_DIR, attr)
        resized_root = os.path.join(settings.ATTR_SAMPLE_DIR, '{}-resized'.format(attr))
        if not os.path.isdir(root):
            os.makedirs(root)
        if not os.path.isdir(resized_root):
            os.makedirs(resized_root)
        for i, (image_id, file_name, bbox, idx_in_img) in enumerate(imgset):
            if i % 100 == 0:
                print(attr, i, '/', len(imgset))
            image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR, file_name))
            cropped_file_path = os.path.join(root, '{}_{}.png'.format(image_id, idx_in_img))
            resized_file_path = os.path.join(resized_root, '{}_{}.png'.format(image_id, idx_in_img))
            cropped = predictions2html.crop(image, bbox, float('inf'))
            resized = cv2.resize(cropped, (32, 32))
            cv2.imwrite(cropped_file_path, cropped)
            cv2.imwrite(resized_file_path, resized)


if __name__ == '__main__':
    main()
