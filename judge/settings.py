DATA_LIST              = '../data/annotations/info.json'
TRAIN                  = '../data/annotations/train.jsonl'
VAL                    = '../data/annotations/val.jsonl'
TEST_CLASSIFICATION_GT = '../data/annotations/test_cls.gt.jsonl'
TEST_DETECTION_GT      = '../data/annotations/test_det.gt.jsonl'
TEST_CLASSIFICATION    = '../data/annotations/test_cls.jsonl'

TRAINVAL_IMAGE_DIR     = '../data/images/trainval'
TEST_IMAGE_DIR         = '../data/images/test'

PRODUCTS_ROOT          = 'products'
DETECTION_EXE          = 'products/detection.bin'
STAT_FREQUENCY         = 'products/stat_frequency.json'
DATASET_CROPPED        = 'products/dataset_cropped.pkl'
TEST_CLS_CROPPED       = 'products/test_cls_cropped.pkl'
PREDICTIONS_HTML       = 'products/predictions_compare.html'
CLASSIFICATION_REPORT  = 'products/explore_cls.html'
DETECTION_EXPLORE      = 'products/explore_det.html'
DETECTION_REPORT       = 'products/detection_report.json'
PROPOSAL_EXPLORE       = 'products/explore_pro.html'
PROPOSAL_REPORT        = 'products/proposal_report.json'
PRINTTEXT_EXEC         = 'products/build-printtext-release/printtext'
PRINTTEXT_DRAWING_DIR  = 'products/printtext-drawing'
ATTR_SAMPLE_DIR        = 'products/attr-samples'
PLOTS_DIR              = 'products/plots'


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
