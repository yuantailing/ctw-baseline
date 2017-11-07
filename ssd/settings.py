DATA_LIST              = '../data/annotations/info.json'
TRAIN                  = '../data/annotations/train.jsonl'
VAL                    = '../data/annotations/val.jsonl'

TRAINVAL_IMAGE_DIR     = '../data/images/trainval'
TEST_IMAGE_DIR         = '../data/images/test'

PRODUCTS_ROOT          = 'products'
CATES                  = 'products/cates.json'
NUM_CHAR_CATES         = 1000
MAX_DET_PER_IMAGE      = 1000

CAFFE_ROOT             = 'caffe'
TRAIN_IMAGE_SIZE       = 300
TRAINVAL_LIST          = 'products/trainval.txt'
VAL_LIST               = 'products/val.txt'
TEST_LIST              = 'products/test.txt'
VAL_NAME_SIZE          = 'products/val_name_size.txt'
LABELMAP_FILE          = 'products/labelmap.prototxt'
PRETRAIN_MODEL         = 'products/VGG_ILSVRC_16_layers_fc_reduced.caffemodel'

TRAINVAL_CROPPED_DIR   = 'products/trainval'
TRAINVAL_LMDB_DIR      = 'products/lmdb_trainval'
VAL_LMDB_DIR           = 'products/lmdb_val'
TEST_CROPPED_DIR       = 'products/test'
TEST_LMDB_DIR          = 'products/lmdb_test'
