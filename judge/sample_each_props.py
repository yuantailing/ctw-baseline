# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import json
import os
import predictions2html
import random
import settings
import six

from collections import defaultdict
from pythonapi import anno_tools
from six.moves import cPickle


def main():
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
            for prop in settings.PROPERTIES:
                if prop in char['properties']:
                    belongs[prop].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
                else:
                    belongs['not-{}'.format(prop)].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
            i += 1

    for prop, imgset in sorted(belongs.items()):
        imgset = random.sample(imgset, 200)
        root = os.path.join(settings.PROPS_SAMPLE_DIR, prop)
        resized_root = os.path.join(settings.PROPS_SAMPLE_DIR, '{}-resized'.format(prop))
        if not os.path.isdir(root):
            os.makedirs(root)
        if not os.path.isdir(resized_root):
            os.makedirs(resized_root)
        for i, (image_id, file_name, bbox, idx_in_img) in enumerate(imgset):
            if i % 100 == 0:
                print(prop, i, '/', len(imgset))
            image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR, file_name))
            cropped_file_path = os.path.join(root, '{}_{}.png'.format(image_id, idx_in_img))
            resized_file_path = os.path.join(resized_root, '{}_{}.png'.format(image_id, idx_in_img))
            cropped = predictions2html.crop(image, bbox, float('inf'))
            resized = cv2.resize(cropped, (32, 32))
            cv2.imwrite(cropped_file_path, cropped)
            cv2.imwrite(resized_file_path, resized)


if __name__ == '__main__':
    main()
