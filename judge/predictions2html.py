# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import json
import math
import numpy as np
import operator
import os
import settings
import six
import sys

from jinja2 import Template
from pythonapi import common_tools
from scipy import misc
from six.moves import cPickle


cfgs = [
    {
        'model_name': 'alexnet_v2',
        'display_name': 'AlexNet',
    },
    {
        'model_name': 'overfeat',
        'display_name': 'OverFeat',
    },
    {
        'model_name': 'inception_v4',
        'display_name': 'Inception',
    },
    {
        'model_name': 'resnet_v2_50',
        'display_name': 'ResNet50',
    },
    {
        'model_name': 'resnet_v2_152',
        'display_name': 'ResNet152',
    },
]

for cfg in cfgs:
    cfg['predictions_file_path'] = os.path.join('..', 'classification', 'products', 'predictions_{}.jsonl'.format(cfg['model_name']))

if 1 < len(sys.argv):
    cfgs = list(filter(lambda o: o['model_name'] in sys.argv[1:], cfgs))


def crop(image, bbox, maxlong):
    expand = 0
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
    if xhi - xlo >= yhi - ylo:
        ylo = round(y + h / 2. - (xhi - xlo) / 2.)
        yhi = ylo + xhi - xlo
    else:
        xlo = round(x + w / 2. - (yhi - ylo) / 2.)
        xhi = xlo + yhi - ylo
    xxlo = max(0, xlo)
    yylo = max(0, ylo)
    xxhi = min(image.shape[1], xhi)
    yyhi = min(image.shape[0], yhi)
    cropped = np.zeros((yhi - ylo, xhi - xlo, 3), dtype=np.uint8) + 128
    assert xxlo < xxhi and yylo < yyhi
    cropped[yylo - ylo:yyhi - ylo, xxlo - xlo:xxhi - xlo, :] = image[yylo:yyhi, xxlo:xxhi].copy()
    if xhi - xlo > maxlong:
        cropped = misc.imresize(cropped, (maxlong, maxlong))
    return cropped


def create_pkl():
    with open(settings.TEST_CLASSIFICATION) as f:
        lines = f.read().splitlines()
    with open(settings.TEST_CLASSIFICATION_GT) as f:
        gt_lines = f.read().splitlines()
    assert len(lines) == len(gt_lines)
    test = []
    for i, line in enumerate(lines):
        anno = json.loads(line.strip())
        gt_anno = json.loads(gt_lines[i].strip())
        image = misc.imread(os.path.join(settings.TEST_IMAGE_DIR, anno['file_name']))
        assert image.shape == (anno['height'], anno['width'], 3)
        assert len(anno['proposals']) == len(gt_anno['ground_truth'])
        for proposal, gt in zip(anno['proposals'], gt_anno['ground_truth']):
            cropped = crop(image, proposal['adjusted_bbox'], 32)
            test.append([cropped, gt])
        if i % 100 == 0:
            print('test', i, '/', len(lines))
    with open(settings.TEST_CLS_CROPPED, 'wb') as f:
        cPickle.dump(test, f)


def main(models, n):
    assert six.PY3

    if not common_tools.exists_and_newer(settings.TEST_CLS_CROPPED, settings.TEST_CLASSIFICATION):
        print('creating', settings.TEST_CLS_CROPPED)
        create_pkl()

    with open(settings.TEST_CLS_CROPPED, 'rb') as f:
        gts = cPickle.load(f)
    preds = []
    for model in models:
        all = []
        with open(model['predictions_file_path']) as f:
            for line in f:
                obj = json.loads(line.strip())
                all += list(zip(obj['predictions'], obj['probabilities']))
        assert len(gts) == len(all)
        preds.append(all)

    np.random.seed(0)
    sampled = np.array(range(len(gts)))
    np.random.shuffle(sampled)
    sampled = sampled[105:105 + n]

    with open('predictions_compare.template.html') as f:
        template = Template(f.read())
    rows = []
    for i in sampled:
        row = dict()
        image, gt = gts[i]
        image = misc.toimage(image)
        bytesio = six.BytesIO()
        image.save(bytesio, format='png')
        png_base64 = base64.b64encode(bytesio.getvalue())

        row['png_base64'] = png_base64.decode('ascii')
        row['ground_truth'] = gt['text']
        row_models = []
        for preds_model in preds:
            texts, probs = preds_model[i]
            prob_text = '{:5.1f}%'.format(round(probs[0] * 1000) / 10.)
            prob_text = prob_text.replace('  ', '&nbsp;')
            row_models.append({
                'text': texts[0],
                'prob': prob_text,
            })
        row['models'] = row_models
        rows.append(row)
    with open(settings.PREDICTIONS_HTML, 'w') as f:
        f.write(template.render({
            'models': list(map(operator.itemgetter('display_name'), models)),
            'rows': rows,
        }))

    with open(settings.STAT_FREQUENCY) as f:
        freq = json.load(f)
    text2idx = {o['text']: i for i, o in enumerate(freq)}

    dir_name = 'cls_examples'
    root = os.path.join(settings.PRODUCTS_ROOT, dir_name)
    if not os.path.isdir(root):
        os.makedirs(root)
    def text2minipage(text):
        return r'\begin{minipage}{3mm} \includegraphics[width=\linewidth]{figure/texts/' \
            + '0_{}.png'.format(text2idx[text]) + r'} \end{minipage}'

    for no, i in enumerate(sampled):
        file_name = '{}.png'.format(i)
        image, gt = gts[i]
        image = misc.toimage(image)
        image.save(os.path.join(root, file_name), format='png')

        s = r'\begin{minipage}{6.0mm} \includegraphics[width=\linewidth]{figure/cls_examples/' + '{}.png'.format(i) + r'} \end{minipage} &' + '\n'
        s += '{} &\n'.format(text2minipage(gt['text']))

        for j, preds_model in enumerate(preds):
            texts, probs = preds_model[i]
            prob_text = '{:5.1f}'.format(round(probs[0] * 1000) / 10.)
            prob_text = prob_text.replace(' ', r'\,\,\,')
            col = '{} {}'.format(text2minipage(texts[0]), prob_text)
            if texts[0] == gt['text']:
                col = r'\multicolumn{1}{>{\columncolor{cls_correct}}r}{' + col + '}'
            if j == len(preds) - 1:
                col += r' \\'
            else:
                col += ' &'
            s += col + '\n'


if __name__ == '__main__':
    main(cfgs, 20)
