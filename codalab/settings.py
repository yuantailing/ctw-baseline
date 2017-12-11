# -*- coding: utf-8 -*-

from __future__ import unicode_literals


DATA_LIST              = '../data/annotations/info.json'
VAL                    = '../data/annotations/val.jsonl'
VAL_CLASSIFICATION_GT  = '../data/annotations/val_cls.gt.jsonl'
TEST_CLASSIFICATION_GT = '../data/annotations/test_cls.gt.jsonl'
TEST_CLASSIFICATION_GT_AES = '../data/annotations/test_cls.gt.jsonl.aes'
TEST_DETECTION_GT      = '../data/annotations/test_det.gt.jsonl'
TEST_DETECTION_GT_AES  = '../data/annotations/test_det.gt.jsonl.aes'

PRODUCTS_ROOT          = 'products'
CODALAB_TEST_REF       = 'products/test_ref'
CODALAB_TEST_OUTPUT    = 'products/test_output'


RECALL_N               = (1, 5)
SIZE_RANGES            = [
    ('all', (0., 4096.)),
    ('large', (32., 4096.)),
    ('medium', (16., 32.)),
    ('small', (0., 16.)),
]
ATTRIBUTES             = [
    'occluded',
    'bgcomplex',
    'distorted',
    'raised',
    'wordart',
    'handwritten',
]
MAX_DET_PER_IMAGE      = 1000
IOU_THRESH             = .5
TOP_CATEGORIES       = set(u'中国大电路车店家公行')

AP_CURVE_MAX_STEP      = 1000
AP_CURVE_MAX_NDIGITS   = 6
