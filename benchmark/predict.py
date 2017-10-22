# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import json
import os
import numpy as np
import settings
import six
import subprocess
import sys
import eval

from pythonapi import common_tools
from six.moves import cPickle


def main(model_name, max_prediction):
    assert six.PY3

    with open(settings.CATES) as f:
        cates = json.load(f)

    cfg_model = list(filter(lambda o: o['model_name'] == model_name, eval.cfgs))[0]
    eval_dir = cfg_model['eval_dir']
    with open(eval_dir, 'rb') as f:
        data = cPickle.load(f)

    for k in sorted(data.keys()):
        if k != 'logits':
            print(k, data[k])
    logits = data['logits']
    assert settings.NUM_CHAR_CATES + 1 == logits.shape[1]
    logits = logits[:, :settings.NUM_CHAR_CATES]
    explogits = np.exp(np.minimum(logits, 70))
    expsums = np.sum(explogits, axis=1)
    expsums.shape = (logits.shape[0], 1)
    expsums = np.repeat(expsums, settings.NUM_CHAR_CATES, axis=1)
    probs = explogits / expsums
    argsorts = np.argsort(-logits, axis=1)

    lo = 0
    pred_file_name = os.path.join(settings.PRODUCTS_ROOT, 'predictions_{}.jsonl'.format(model_name))
    with open(settings.TEST_CLASSIFICATION) as f, open(pred_file_name, 'w') as fout:
        for line in f:
            image_info = json.loads(line.strip())
            n = len(image_info['proposals'])
            predictions = []
            probabilities = []
            for i in range(n):
                pred = argsorts[lo][:max_prediction]
                prob = probs[lo][pred].tolist()
                pred = list(map(lambda i: cates[i]['text'], pred.tolist()))
                predictions.append(pred)
                probabilities.append(prob)
                lo += 1
            fout.write(common_tools.to_jsonl({
                'predictions': predictions,
                'probabilities': probabilities,
            }))
            fout.write('\n')
    assert lo == logits.shape[0]

if __name__ == '__main__':
    main(sys.argv[1], 5)
