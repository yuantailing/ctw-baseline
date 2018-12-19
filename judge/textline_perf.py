# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import settings

from pythonapi import eval_tools


def main():
    with open(settings.TEST_DETECTION_GT) as f:
        gt = f.read()
    with open('../detection/products/textlines.jsonl') as f:
        dt = f.read()
    report = eval_tools.textline_AED(gt, dt, settings.TEXTLINE_MAX_DET, settings.TEXTLINE_MAX_VERTICES, settings.TEXTLINE_IOU_THRESH, echo=True)
    print(report)


if __name__ == '__main__':
    main()
