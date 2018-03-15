# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
import os
import predictions2html
import settings
import six

from pythonapi import common_tools
from scipy import misc
from six.moves import cPickle


def main(models, n):
    assert six.PY3

    if not common_tools.exists_and_newer(settings.TEST_CLS_CROPPED, settings.TEST_CLASSIFICATION):
        print('creating', settings.TEST_CLS_CROPPED)
        predictions2html.create_pkl()

    with open(settings.TEST_CLS_CROPPED, 'rb') as f:
        gts = cPickle.load(f)
    with open(settings.STAT_FREQUENCY) as f:
        stat_freq = json.load(f)
    preds = []
    for model in models:
        all = []
        with open(model['predictions_file_path']) as f:
            for line in f:
                obj = json.loads(line.strip())
                all += list(zip(obj['predictions'], obj['probabilities']))
        assert len(gts) == len(all)
        preds.append(all)

    np.random.seed(n + 2018)
    sampled = np.array(range(len(gts)))
    np.random.shuffle(sampled)
    sampled = sampled[:n]

    dir_name = 'cls_examples'
    root = os.path.join(settings.PRODUCTS_ROOT, dir_name)
    if not os.path.isdir(root):
        os.makedirs(root)

    def text2minipage(text):
        i = [o['text'] for o in stat_freq].index(text)
        return r'\begin{minipage}{3.5mm} \includegraphics[width=\linewidth]{figure/texts/' + '0_{}.png'.format(i) + r'} \end{minipage}'

    for no, i in enumerate(sampled):
        file_name = '{}.png'.format(i)
        image, gt = gts[i]
        image = misc.toimage(image)
        image.save(os.path.join(root, file_name), format='png')

        s = '{} & '.format(no + 1) + r'\begin{minipage}{6.0mm} \includegraphics[width=\linewidth]{figure/cls_examples/' + '{}.png'.format(i) + r'} \end{minipage} &' + '\n'
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
        print(s.replace('\n', ' ').strip())


if __name__ == '__main__':
    main(predictions2html.cfgs, 50)
