# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import darknet_tools
import json
import numpy as np
import os
import settings

from jinja2 import Template
from pythonapi import anno_tools, common_tools
from six.moves import queue


def write_darknet_data():
    if not os.path.exists(settings.DARKNET_BACKUP_DIR):
        os.makedirs(settings.DARKNET_BACKUP_DIR)
    if not os.path.exists(settings.DARKNET_RESULTS_DIR):
        os.makedirs(settings.DARKNET_RESULTS_DIR)
    data = {
        'classes': settings.NUM_CHAR_CATES + 1,
        'train': settings.DARKNET_TRAIN_LIST,
        'valid': settings.DARKNET_VALID_LIST,
        'names': settings.DARKNET_NAMES,
        'backup': settings.DARKNET_BACKUP_DIR,
        'results': settings.DARKNET_RESULTS_DIR,
    }
    with open(settings.DARKNET_DATA, 'w') as f:
        for k, v in sorted(data.items()):
            f.write('{} = {}\n'.format(k, v))


def write_darknet_cfg():
    with open('yolo-chinese.template.cfg') as f:
        template = Template(f.read())
    with open(settings.DARKNET_CFG, 'w') as f:
        f.write(template.render({
            'testing': False,
            'image_size': settings.TRAIN_IMAGE_SIZE,
            'classes': settings.NUM_CHAR_CATES + 1,
            'filters': 25 + 5 * (settings.NUM_CHAR_CATES + 1),
        }))
        f.write('\n')


def write_darknet_names():
    with open(settings.DARKNET_NAMES, 'w') as f:
        for i in range(settings.NUM_CHAR_CATES + 1):
            f.write('{}\n'.format(i))


def crop_train_images():
    imshape = (2048, 2048, 3)
    cropshape = (settings.TRAIN_IMAGE_SIZE // 4, settings.TRAIN_IMAGE_SIZE // 4)
    cropoverlap = (16, 16)

    with open(settings.CATES) as f:
        cates = json.load(f)
    text2cate = {c['text']: c['cate_id'] for c in cates}

    if not os.path.isdir(settings.TRAINVAL_CROPPED_DIR):
        os.makedirs(settings.TRAINVAL_CROPPED_DIR)

    with open(settings.TRAIN) as f:
        lines = f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()

    def in_image_ratio(bbox):  # bbox is in darknet bbox representation
        xmid, ymid, w, h = bbox

        def cutto01(x):
            return max(0, min(1, x))
        Acut = (cutto01(xmid + w / 2) - cutto01(xmid - w / 2)) * (cutto01(ymid + h / 2) - cutto01(ymid - h / 2))
        return Acut / (w * h)

    def crop_once(line, write_images):
        anno = json.loads(line.strip())
        image_id = anno['image_id']
        all = []
        for char in anno_tools.each_char(anno):
            if not char['is_chinese']:
                continue
            cate_id = text2cate[char['text']]
            if cate_id >= settings.NUM_CHAR_CATES:
                cate_id = settings.NUM_CHAR_CATES
            all.append((char['adjusted_bbox'], cate_id))
        if write_images:
            image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR, anno['file_name']))
            assert image.shape == imshape
            for o in anno['ignore']:
                poly = (np.array(o['polygon'])).astype(np.int32)
                cv2.fillConvexPoly(image, poly, (128, 128, 128))
        cropped_list = list()
        for o in darknet_tools.get_crop_bboxes(imshape, cropshape, cropoverlap):
            xlo = o['xlo']
            xhi = xlo + cropshape[1]
            ylo = o['ylo']
            yhi = ylo + cropshape[0]
            labels = []
            for bbox, cate_id in all:
                x, y, w, h = bbox
                if x > xhi or x + w < xlo or y > yhi or y + h < ylo:
                    continue
                bbox = ((x + w / 2 - xlo) / cropshape[1], (y + h / 2 - ylo) / cropshape[0], w / cropshape[1], h / cropshape[0])
                if 0.5 < in_image_ratio(bbox):
                    labels.append((bbox, cate_id))
            if 0 < len(labels):
                basename = '{}_{}'.format(image_id, o['name'])
                cropped_file_name = os.path.join(settings.TRAINVAL_CROPPED_DIR, '{}.jpg'.format(basename))
                cropped_list.append(cropped_file_name)
                if write_images:
                    cropped = image[ylo:yhi, xlo:xhi]
                    cv2.imwrite(cropped_file_name, cropped)
                    with open(os.path.join(settings.TRAINVAL_CROPPED_DIR, '{}.txt'.format(basename)), 'w') as f:
                        for bbox, cate_id in labels:
                            f.write('%d %f %f %f %f\n' % ((cate_id, ) + bbox))
        return cropped_list

    q_i = queue.Queue()
    q_i.put(0)

    def foo(*args):
        i = q_i.get()
        if i % 100 == 0:
            print('crop trainval', i, '/', len(lines))
        q_i.put(i + 1)
        crop_once(*args)
    common_tools.multithreaded(foo, [(line, True) for line in lines], num_thread=4)
    trainset = []
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print('list trainval', i, '/', len(lines))
        trainset += crop_once(line, False)
    with open(settings.DARKNET_TRAIN_LIST, 'w') as f:
        for file_name in trainset:
            f.write(file_name)
            f.write('\n')


def main():
    write_darknet_data()
    write_darknet_cfg()
    write_darknet_names()
    assert os.path.isfile(settings.DARKNET_PRETRAIN) and 79327120 == os.path.getsize(settings.DARKNET_PRETRAIN), \
            'please download {} to {}'.format('https://pjreddie.com/media/files/darknet19_448.conv.23', settings.DARKNET_PRETRAIN)
    if not common_tools.exists_and_newer(settings.DARKNET_TRAIN_LIST, settings.CATES):
        crop_train_images()


if __name__ == '__main__':
    main()
