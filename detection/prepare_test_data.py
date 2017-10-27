# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import darknet_tools
import json
import os
import settings

from jinja2 import Template
from pythonapi import common_tools
from six.moves import queue


def write_darknet_test_cfg():
    with open('yolo-chinese.template.cfg') as f:
        template = Template(f.read())
    with open(settings.DARKNET_TEST_CFG, 'w') as f:
        f.write(template.render({
            'testing': True,
            'image_size': settings.TEST_IMAGE_SIZE,
            'classes': settings.NUM_CHAR_CATES + 1,
            'filters': 25 + 5 * (settings.NUM_CHAR_CATES + 1),
        }))
        f.write('\n')


def crop_test_images():
    imshape = (2048, 2048, 3)

    with open(settings.CATES) as f:
        cates = json.load(f)
    text2cate = {c['text']: c['cate_id'] for c in cates}

    if not os.path.isdir(settings.TEST_CROPPED_DIR):
        os.mkdir(settings.TEST_CROPPED_DIR)

    with open(settings.DATA_LIST) as f:
        data_list = json.load(f)
    test_cls = data_list['test_cls']

    def crop_once(anno, write_images):
        image_id = anno['image_id']
        if write_images:
            image = cv2.imread(os.path.join(settings.TEST_IMAGE_DIR, anno['file_name']))
            assert image.shape == imshape
        cropped_list = []
        for level_id, (cropratio, cropoverlap) in enumerate(settings.TEST_CROP_LEVELS):
            cropshape = (settings.TEST_IMAGE_SIZE // cropratio, settings.TEST_IMAGE_SIZE // cropratio)
            for o in darknet_tools.get_crop_bboxes(imshape, cropshape, (cropoverlap, cropoverlap)):
                xlo = o['xlo']
                xhi = xlo + cropshape[1]
                ylo = o['ylo']
                yhi = ylo + cropshape[0]
                basename = '{}_{}_{}'.format(image_id, level_id, o['name'])
                cropped_file_name = os.path.join(settings.TEST_CROPPED_DIR, '{}.jpg'.format(basename))
                cropped_list.append(cropped_file_name)
                if write_images:
                    cropped = image[ylo:yhi, xlo:xhi]
                    cv2.imwrite(cropped_file_name, cropped)
        return cropped_list

    q_i = queue.Queue()
    q_i.put(0)
    def foo(*args):
        i = q_i.get()
        if i % 100 == 0:
            print('crop test', i, '/', len(test_cls))
        q_i.put(i + 1)
        crop_once(*args)
    common_tools.multithreaded(foo, [(anno, True) for anno in test_cls], num_thread=4)
    testset = []
    for i, anno in enumerate(test_cls):
        if i % 1000 == 0:
            print('list test', i, '/', len(test_cls))
        testset += crop_once(anno, False)
    with open(settings.DARKNET_VALID_LIST, 'w') as f:
        for file_name in testset:
            f.write(file_name)
            f.write('\n')


def main():
    write_darknet_test_cfg()
    if not common_tools.exists_and_newer(settings.DARKNET_VALID_LIST, settings.CATES):
        crop_test_images()


if __name__ == '__main__':
    main()
