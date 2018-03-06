DATA_LIST              = '../data/annotations/info.json'
TRAIN                  = '../data/annotations/train.jsonl'
VAL                    = '../data/annotations/val.jsonl'

TRAINVAL_IMAGE_DIR     = '../data/images/trainval'
TEST_IMAGE_DIR         = '../data/images/test'

PRODUCTS_ROOT          = 'products'
CATES                  = 'products/cates.json'
NUM_CHAR_CATES         = 1000
MAX_DET_PER_IMAGE      = 1000

DARKNET_ROOT           = 'darknet'
DARKNET_BACKUP_DIR     = 'products/backup'
DARKNET_RESULTS_DIR    = 'products/results'
DARKNET_RESULTS_OUT    = 'chinese'

DARKNET_DATA           = 'products/chinese.data'
DARKNET_CFG            = 'products/yolo-chinese.cfg'
DARKNET_TEST_CFG       = 'products/yolo-chinese-test.cfg'
DARKNET_NAMES          = 'products/chinese.names'
DARKNET_PRETRAIN       = 'products/darknet19_448.conv.23'

DARKNET_TRAIN_LIST     = 'products/trainval.txt'
DARKNET_VALID_LIST     = 'products/test.txt'

TRAIN_IMAGE_SIZE       = 672
TEST_IMAGE_SIZE        = 1216
TEST_SPLIT_NUM         = 12
TEST_NUM_GPU           = 2
TEST_CROP_LEVELS       = ((4, 32), (2, 96), (1, 96))
TRAINVAL_CROPPED_DIR   = 'products/trainval'
TEST_CROPPED_DIR       = 'products/test'
