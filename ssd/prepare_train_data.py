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
import subprocess
import xml.dom.minidom

from pythonapi import anno_tools, common_tools
from six.moves import queue


def write_xml(labels, cropshape, f):
    impl = xml.dom.minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'annotation', None)
    annotation = dom.documentElement
    size = dom.createElement('size')
    width = dom.createElement('width')
    height = dom.createElement('height')
    depth = dom.createElement('depth')
    widthText = dom.createTextNode(str(cropshape[1]))
    heightText = dom.createTextNode(str(cropshape[0]))
    depthText = dom.createTextNode(str(3))
    annotation.appendChild(size)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    width.appendChild(widthText)
    height.appendChild(heightText)
    depth.appendChild(depthText)
    segmented = dom.createElement('segmented')
    segmentedText = dom.createTextNode(str(0))
    annotation.appendChild(segmented)
    segmented.appendChild(segmentedText)
    for bbox, cate_id in labels:
        label, xmid, ymid, w, h = str(cate_id), bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = max(0, min(cropshape[1] - 1, int((xmid - w / 2) * cropshape[1])))
        x2 = max(0, min(cropshape[1] - 1, int((xmid + w / 2) * cropshape[1])))
        y1 = max(0, min(cropshape[0] - 1, int((ymid - h / 2) * cropshape[0])))
        y2 = max(0, min(cropshape[0] - 1, int((ymid + h / 2) * cropshape[0])))
        if x1 == x2 or y1 == y2:
            continue
        obj = dom.createElement('object')
        name = dom.createElement('name')
        bndbox = dom.createElement('bndbox')
        xmin = dom.createElement('xmin')
        ymin = dom.createElement('ymin')
        xmax = dom.createElement('xmax')
        ymax = dom.createElement('ymax')
        nameText = dom.createTextNode(label)
        xminText = dom.createTextNode(str(x1))
        yminText = dom.createTextNode(str(y1))
        xmaxText = dom.createTextNode(str(x2))
        ymaxText = dom.createTextNode(str(y2))
        annotation.appendChild(obj)
        obj.appendChild(name)
        obj.appendChild(bndbox)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
        name.appendChild(nameText)
        xmin.appendChild(xminText)
        ymin.appendChild(yminText)
        xmax.appendChild(xmaxText)
        ymax.appendChild(ymaxText)
    dom.writexml(f, addindent='\t', newl='\n')


def crop_train_images():
    imshape = (2048, 2048, 3)
    cropshape = (settings.TRAIN_IMAGE_SIZE, settings.TRAIN_IMAGE_SIZE)
    cropoverlap = (16, 16)

    with open(settings.CATES) as f:
        cates = json.load(f)
    text2cate = {c['text']: c['cate_id'] for c in cates}

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
                if 0.7 < in_image_ratio(bbox):
                    labels.append((bbox, cate_id))
            if 0 < len(labels):
                basename = '{}_{}'.format(image_id, o['name'])
                cropped_file_name = os.path.join(settings.TRAINVAL_CROPPED_DIR, '{}.jpg'.format(basename))
                cropped_xml_name = os.path.join(settings.TRAINVAL_CROPPED_DIR, '{}.xml'.format(basename))
                cropped_list.append('{} {}'.format(cropped_file_name, cropped_xml_name))
                if write_images:
                    cropped = image[ylo:yhi, xlo:xhi]
                    cv2.imwrite(cropped_file_name, cropped)
                    with open(os.path.join(settings.TRAINVAL_CROPPED_DIR, '{}.xml'.format(basename)), 'w') as f:
                        write_xml(labels, cropshape, f)
        return cropped_list

    if not os.path.isdir(settings.TRAINVAL_CROPPED_DIR):
        os.makedirs(settings.TRAINVAL_CROPPED_DIR)

    lines = []
    with open(settings.TRAIN) as f:
        lines += f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()

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
    with open(settings.TRAINVAL_LIST, 'w') as f:
        for file_name in trainset:
            f.write(file_name)
            f.write('\n')

    with open(settings.VAL) as f:
        lines = f.read().splitlines()
    valset = []
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print('list val', i, '/', len(lines))
        valset += crop_once(line, False)
    with open(settings.VAL_LIST, 'w') as f:
        for file_name in trainset:
            f.write(file_name)
            f.write('\n')
    with open(settings.VAL_NAME_SIZE, 'w') as f:
        for line in valset:
            cropped_file_name, cropped_xml_name = line.split()
            f.write('{} {} {}\n'.format(cropped_file_name, settings.TRAIN_IMAGE_SIZE, settings.TRAIN_IMAGE_SIZE))


def write_caffe_labelmap():
    with open(settings.LABELMAP_FILE, 'w') as f:
        for i in range(-1, settings.NUM_CHAR_CATES + 1):
            name = 'background' if i == -1 else str(i)
            f.write('item {\n')
            f.write('  name: "{:s}"\n'.format(name))
            f.write('  label: {:d}\n'.format(i + 1))
            f.write('  display_name: "{:s}"\n'.format(name))
            f.write('}\n')


def create_lmdb():
    exe = 'python2'
    script = os.path.join('ssd_hardcode', 'create_annoset.py')
    kw = {
        'anno-type': 'detection',
        'label-map-file': settings.LABELMAP_FILE,
        'min-dim': 0,
        'max-dim': 0,
        'resize-width': 0,
        'resize-height': 0,
        'check-label': None,
        'encode-type': 'jpg',
        'encoded': None,
        'redo': None,
    }
    args = [exe, script]
    for k, v in sorted(kw.items()):
        if v is None:
            args.append('--{}'.format(k))
        else:
            args.append('--{}={}'.format(k, v))
    args_0 = args + ['.', settings.TRAINVAL_LIST, settings.TRAINVAL_LMDB_DIR]
    args_1 = args + ['.', settings.VAL_LIST, settings.VAL_LMDB_DIR]

    env = os.environ.copy()
    pythonpath = os.path.join(settings.CAFFE_ROOT, 'python')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = '{}:{}:{}'.format(env['PYTHONPATH'], '.', pythonpath)
    else:
        env['PYTHONPATH'] = '{}:{}'.format('.', pythonpath)

    print(*args_0)
    p = subprocess.Popen(args_0, env=env)
    p.wait()
    assert 0 == p.returncode

    print(*args_1)
    p = subprocess.Popen(args_1, env=env)
    p.wait()
    assert 0 == p.returncode


def main():
    assert os.path.isfile(os.path.join(settings.CAFFE_ROOT, 'python', 'caffe', '_caffe.so')), 'please compile pycaffe'
    crop_train_images()
    write_caffe_labelmap()
    create_lmdb()


if __name__ == '__main__':
    main()
