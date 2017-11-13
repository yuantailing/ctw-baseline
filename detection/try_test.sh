#!/bin/bash
set -e
sh -c "cd darknet && make -j12"
#darknet/darknet detector test products/chinese.data products/yolo-chinese-test.cfg ../detection/products/backup/yolo-chinese_24000.weights products/test/0000005_1_2_0.jpg -thresh 0.3
darknet/darknet detector test products/chinese.data products/yolo-chinese-test.cfg ../detection/products/backup/yolo-chinese_24000.weights products/test/0000339_1_2_2.jpg -thresh 0.1
#darknet/darknet detector test products/chinese.data products/yolo-chinese.cfg ../../bak_ch_CODE0/task_3_engine/backup/yolo-character_final.weights products/trainval/3001243_7_3.jpg -thresh 0.1
