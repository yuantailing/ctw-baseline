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
    with open(settings.TRAIN) as f:
        for line in f:
            anno = json.loads(line.strip())
            for char in anno_tools.each_char(anno):
                if char['is_chinese']:
                    text = char['text']
                    assert 1 == len(text)
                    counts[text] += 1
    assert settings.NUM_CHAR_CATES <= len(counts)
    a = sorted([(v, k) for k, v in counts.items()])
    not_ordered = [t[1] for t in a[-settings.NUM_CHAR_CATES:]]
    ordered = sorted(not_ordered, key=lambda x: x.encode('cp936'))
    cates = [{'cate_id': i, 'text': k, 'train': counts[k]} for i, k in enumerate(ordered)]
    with open(settings.CATES, 'w') as f:
        json.dump(cates, f, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
