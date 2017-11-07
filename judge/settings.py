DATA_LIST              = '../data/annotations/info.json'
TRAIN                  = '../data/annotations/train.jsonl'
VAL                    = '../data/annotations/val.jsonl'
TEST_CLASSIFICATION_GT = '../data/annotations/test_cls.gt.jsonl'
TEST_DETECTION_GT      = '../data/annotations/test_det.gt.jsonl'
TEST_CLASSIFICATION    = '../data/annotations/test_cls.jsonl'

TRAINVAL_IMAGE_DIR     = '../data/images/trainval'
TEST_IMAGE_DIR         = '../data/images/test'

PRODUCTS_ROOT          = 'products'
TEST_CLS_CROPPED       = 'products/test_cls_cropped.pkl'
PREDICTIONS_HTML       = 'products/predictions_compare.html'
CLASSIFICATION_REPORT  = 'products/explore_cls.html'
DETECTION_REPORT       = 'products/detection_report.json'
PRINTTEXT_EXEC         = 'products/build-printtext-release/printtext'
PRINTTEXT_DRAWING_DIR  = 'products/printtext-drawing'
PROPS_SAMPLE_DIR       = 'products/props-samples'


RECALL_N               = (1, 5)
SIZE_RANGES            = [
    ('__all__', (0., float('inf'))),
    ('small(<16)', (0., 16.)),
    ('medium', (16., 32.)),
    ('large(>=32)', (32., float('inf'))),
]
PROPERTIES             = [
    'covered',
    'bgcomplex',
    'distorted',
    'raised',
    'wordart',
    'handwritten',
]
MAX_DET_PER_IMAGE      = 1000
IOU_THRESH             = .5
