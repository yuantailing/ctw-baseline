# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import settings
import shutil

from pythonapi import anno_tools, common_tools


def main():
    # load from downloads
    with open('../data/annotations/downloads/info.json') as f:
        full_data_list = json.load(f)
    with open('../data/annotations/downloads/val.jsonl') as f:
        val = f.read().splitlines()
        
    # make infomation of data list
    data_list = {
        'train': full_data_list['train'],
        'val': [],
        'test_cls': full_data_list['val'],
        'test_det': full_data_list['val'],
    }
    with open(settings.DATA_LIST, 'w') as f:
        json.dump(data_list, f, indent=2)

    # copy training set
    shutil.copy('../data/annotations/downloads/train.jsonl', settings.TRAIN)

    # create empty validation set
    with open(settings.VAL, 'w') as f:
        pass

    # create testing set for classification
    with open(settings.TEST_CLASSIFICATION, 'w') as f, open(settings.TEST_CLASSIFICATION_GT, 'w') as fgt:
        for line in val:
            anno = json.loads(line.strip())
            proposals = []
            gt = []
            for char in anno_tools.each_char(anno):
                if not char['is_chinese']:
                    continue
                proposals.append({'adjusted_bbox': char['adjusted_bbox'], 'polygon': char['polygon']})
                gt.append({'text': char['text'], 'attributes': char['attributes'], 'size': char['adjusted_bbox'][-2:]})
            anno.pop('annotations')
            anno.pop('ignore')
            anno['proposals'] = proposals
            f.write(common_tools.to_jsonl(anno))
            f.write('\n')
            anno.pop('proposals')
            anno['ground_truth'] = gt
            fgt.write(common_tools.to_jsonl(anno))
            fgt.write('\n')

    # create testing set for detection
    shutil.copy('../data/annotations/downloads/val.jsonl', settings.TEST_DETECTION_GT)


if __name__ == '__main__':
    main()
