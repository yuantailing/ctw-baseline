# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
import settings

from collections import defaultdict
from six.moves import cPickle
from pythonapi import anno_tools


def main():
    np.random.seed(0)
    counts = defaultdict(lambda: 0)
    with open(settings.TRAIN) as f, open(settings.VAL) as f2:
        for line in f.read().splitlines() + f2.read().splitlines():
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    text = char['text']
                    assert 1 == len(text)
                    counts[text] += 1
    a = sorted([(-v, k) for k, v in counts.items()])
    ordered = [t[1] for t in a]
    cates = [{'cate_id': i, 'text': k, 'trainval': counts[k]} for i, k in enumerate(ordered)]
    with open(settings.CATES, 'w') as f:
        json.dump(cates, f, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
