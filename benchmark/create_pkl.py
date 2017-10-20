# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import math
import numpy as np
import settings
import six

from scipy import misc
from six.moves import cPickle
from pythonapi import anno_tools, common_tools


def crop(image, bbox):
    expand = 0.1
    maxlong = 64
    x, y, w, h = bbox
    x -= w * expand
    w += w * expand * 2
    y -= h * expand
    h += h * expand * 2
    xlo = int(math.floor(x))
    ylo = int(math.floor(y))
    xhi = int(math.ceil(x + w))
    yhi = int(math.ceil(y + h))
    assert xlo < xhi and ylo < yhi
    xxlo = max(0, xlo)
    yylo = max(0, ylo)
    xxhi = min(image.shape[1], xhi)
    yyhi = min(image.shape[0], yhi)
    cropped = np.zeros((yhi - ylo, xhi - xlo, 3), dtype=np.uint8) + 128
    assert xxlo < xxhi and yylo < yyhi
    cropped[yylo - ylo:yyhi - ylo, xxlo - xlo:xxhi - xlo, :] = image[yylo:yyhi, xxlo:xxhi].copy()
    if xhi - xlo >= yhi - ylo:
        if xhi - xlo > maxlong:
            cropped = misc.imresize(cropped, (int(math.ceil(maxlong / (xhi - xlo) * (yhi - ylo))), maxlong))
    else:
        if yhi - ylo > maxlong:
            cropped = misc.imresize(cropped, (maxlong, int(math.ceil(maxlong / (yhi - ylo) * (xhi - xlo)))))
    return cropped


def main():
    assert six.PY3
    with open(settings.CATES) as f:
        cates = json.load(f)
    text2cate = {c['text']: c['cate_id'] for c in cates}

    with open(settings.TRAIN) as f:
        lines = f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()
    train = []
    for i, line in enumerate(lines):
        anno = json.loads(line.strip())
        image = misc.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR, anno['file_name']))
        assert image.shape == (anno['height'], anno['width'], 3)
        for char in anno_tools.each_char(anno):
            if not char['is_chinese']:
                continue
            cate_id = text2cate.get(char['text'])
            cropped = crop(image, char['adjusted_bbox'])
            train.append([cropped, cate_id])
        if i % 100 == 0:
            print('trainval', i, '/', len(lines))
    with open(settings.TRAINVAL_PICKLE, 'wb') as f:
        cPickle.dump(train, f)
    train = [] # release memory

    with open(settings.TEST_CLASSIFICATION) as f:
        lines = f.read().splitlines()
    test = []
    for i, line in enumerate(lines):
        anno = json.loads(line.strip())
        image = misc.imread(os.path.join(settings.TEST_IMAGE_DIR, anno['file_name']))
        assert image.shape == (anno['height'], anno['width'], 3)
        for char in anno['proposals']:
            cropped = crop(image, char['adjusted_bbox'])
            test.append([cropped, None])
        if i % 100 == 0:
            print('test', i, '/', len(lines))
    with open(settings.TEST_PICKLE, 'wb') as f:
        cPickle.dump(test, f)


if __name__ == '__main__':
    main()
