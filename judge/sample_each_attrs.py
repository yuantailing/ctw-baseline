# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import json
import numpy as np
import os
import predictions2html
import random
import settings
import six

from collections import defaultdict
from pythonapi import anno_tools, common_tools
from six.moves import cPickle


def get_polygons():
    lines = []
    with open(settings.TRAIN) as f:
        lines += [('train', s) for s in f.read().splitlines()]
    with open(settings.VAL) as f:
        lines += [('val', s) for s in f.read().splitlines()]
    with open(settings.TEST_CLASSIFICATION) as f1, open(settings.TEST_CLASSIFICATION_GT) as f2:
        prs = f1.read().splitlines()
        gts = f2.read().splitlines()
        lines += [('test_cls', pr, gt) for pr, gt in zip(prs, gts)]
    with open(settings.TEST_DETECTION_GT) as f:
        lines += [('test_det', s) for s in f.read().splitlines()]
    polygons = []
    for i, line in enumerate(lines):
        if line[0] == 'test_cls':
            prs = line[1]
            prs = json.loads(prs.strip())
            for pr in prs['proposals']:
                polygons.append(pr['polygon'])
        else:
            anno = json.loads(line[1].strip())
            for char in anno_tools.each_char(anno):
                if not char['is_chinese']:
                    continue
                polygons.append(char['polygon'])
    return polygons


def polygon_in_center(polygon, removal):
    xy = list(zip(*polygon))
    if min(xy[0]) < removal or max(xy[0]) >= 2048 - removal or min(xy[1]) < removal or max(xy[1]) >= 2048 - removal:
        return False
    return True


def main():
    assert six.PY3
    random.seed(0)

    polygons = get_polygons()
    print('polygons loaded')

    if not os.path.isfile(settings.DATASET_CROPPED):
        lines = []
        with open(settings.TRAIN) as f:
            lines += [('train', s) for s in f.read().splitlines()]
        with open(settings.VAL) as f:
            lines += [('val', s) for s in f.read().splitlines()]
        with open(settings.TEST_CLASSIFICATION) as f1, open(settings.TEST_CLASSIFICATION_GT) as f2:
            prs = f1.read().splitlines()
            gts = f2.read().splitlines()
            lines += [('test_cls', pr, gt) for pr, gt in zip(prs, gts)]
        with open(settings.TEST_DETECTION_GT) as f:
            lines += [('test_det', s) for s in f.read().splitlines()]
        all = [[] for _ in lines]
        def load_train(i):
            if i % 100 == 0:
                print('dataset', i, '/', len(lines))
            line = lines[i]
            if line[0] == 'test_cls':
                prs, gts = line[1:]
                prs, gts = json.loads(prs), json.loads(gts)
                image = cv2.imread(os.path.join(settings.TEST_IMAGE_DIR, prs['file_name']))
                assert image.shape == (prs['height'], prs['width'], 3)
                for pr, gt in zip(prs['proposals'], gts['ground_truth']):
                    cropped = predictions2html.crop(image, pr['adjusted_bbox'], 64)
                    all[i].append([cropped, gt['attributes']])
            else:
                anno = json.loads(line[1].strip())
                image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR if line[0] in {'train', 'val'} else settings.TEST_IMAGE_DIR, anno['file_name']))
                assert image.shape == (anno['height'], anno['width'], 3)
                for char in anno_tools.each_char(anno):
                    if not char['is_chinese']:
                        continue
                    cropped = predictions2html.crop(image, char['adjusted_bbox'], 64)
                    all[i].append([cropped, char['attributes']])
        common_tools.multithreaded(load_train, range(len(lines)), num_thread=8)
        all = common_tools.reduce_sum(all)
        with open(settings.DATASET_CROPPED, 'wb') as f:
            cPickle.dump(all, f)

    with open(settings.DATASET_CROPPED, 'rb') as f:
        all = cPickle.load(f)
    print('cropped loaded')

    belongs = defaultdict(list)
    for i, (_, attrs) in enumerate(all):
        for attr in settings.ATTRIBUTES:
            if attr in attrs:
                belongs[attr].append(i)
            else:
                belongs['not-{}'.format(attr)].append(i)
    x, y, cropsize = 25, 20, 64
    forbidden = {
        'bgcomplex': {29784, 30793, 54066, 60905, 80018, 85910, 92040, 93009, 108277, 117829, 145939, 159277, 166330, 166891, 168897, 174142, 181156, 181461, 185076, 197249, 197278, 197390, 197483, 197736, 233084, 241839, 267090, 278862, 282057, 304974, 305250, 309420, 309505, 311269, 312195, 317930, 318505, 366441, 366568, 366798, 367485, 367889, 369698, 371721, 372093, 372993, 373129, 378209, 438060, 438222, 451115, 451181, 452219, 476071, 494493, 511841, 537930, 568693, 591312, 593805, 604174, 604482, 607029, 613924, 620551, 624535, 630988, 644721, 662645, 685127, 692149, 697909, 713403, 718308, 722478, 731465, 744738, 752377, 756651, 756983, 757500, 838142, 840193, 853782, 868967, 870828, 895909, 897855, 914324, 924949, 926112, 948187, 949547, 951922, 966718, 990845},
        'distorted': {4272, 11545, 31628, 66623, 77833, 80068, 82815, 101418, 101429, 101461, 101475, 102750, 103990, 106226, 122330, 122629, 122864, 127239, 137058, 138722, 165570, 215740, 215956, 241371, 244889, 300272, 311629, 351641, 354914, 355965, 407871, 493919, 516472, 520137, 531795, 545332, 560774, 564422, 568087, 568530, 580687, 580940, 584408, 584961, 587857, 595353, 605587, 635609, 646782, 659324, 674405, 676416, 677631, 763798, 776019, 794110, 799670, 799845, 817479, 827616, 891523, 903911, 909307, 925630, 938309, 942272, 951543, 971224, 974709, 993244, 999040, 1000422},
        'handwritten': {329601, 408868, 512096, 821406, 978196, 982861, 982872, 982875, 1008997},
        'not-bgcomplex': {9246, 15606, 16655, 36299, 61621, 68809, 262313, 270993, 272949, 282797, 310460, 314072, 352096, 399965, 403986, 405162, 606926, 677085, 703288, 779237, 822430, 827693, 850763, 875473, 915644, 922101, 933638, 938877, 990631, 996811, 1010739},
        'not-distorted': {89019, 288330, 289995, 303808, 316933, 420413, 489284, 534030, 585589, 590783, 639562, 652671, 687953, 774776, 845746, 886599, 887318, 955774, 957490},
        'not-handwritten': set(),
        'not-occluded': {4608, 5314, 209656, 242426, 288072, 323822, 434571, 500784, 567581, 569271, 666036, 693361, 854716},
        'not-raised': {32737, 196852, 301792, 320325, 476295, 534636, 652281, 704042, 910982, 915965, 950785},
        'not-wordart': {3645, 50260, 50296, 57502, 82294, 96621, 109453, 204976, 262595, 269395, 284668, 350879, 382009, 414268, 509196, 513846, 516881, 524924, 557900, 567024, 582488, 644754, 647845, 670733, 672294, 683110, 685146, 697891, 704277, 711269, 718931, 731280, 734266, 757216, 792544, 805942, 806092, 814184, 821115, 826304, 836793, 881432, 882749, 882764, 887005, 890080, 898900, 918417, 941640, 944208},
        'occluded': {873, 5730, 8228, 13395, 15541, 41004, 47143, 51186, 61060, 74182, 123631, 124450, 135852, 147417, 157442, 172524, 184918, 185294, 190257, 190412, 190643, 192596, 197677, 224041, 226001, 227348, 227590, 230286, 232497, 235413, 245273, 246929, 248790, 252125, 257097, 272903, 272904, 277566, 284177, 284181, 306556, 309870, 310208, 310823, 312420, 313796, 315943, 320599, 324504, 325094, 345861, 348978, 350766, 355099, 355197, 361102, 363122, 364891, 371829, 375936, 378402, 383816, 385305, 385587, 406896, 429101, 441511, 457760, 460850, 464921, 532963, 532972, 537419, 537840, 566316, 567464, 570925, 575854, 576324, 580475, 580786, 582479, 587189, 601068, 612825, 627320, 629511, 645262, 648763, 660725, 670146, 671016, 676628, 702305, 728385, 734681, 735671, 745464, 747664, 767248, 778261, 779184, 779315, 781154, 786504, 786792, 789390, 797219, 799166, 810608, 836407, 837725, 843622, 843863, 851364, 864668, 868260, 870504, 872464, 888636, 892626, 939872, 940036, 941901, 956912, 976644, 979918, 992558, 1000251, 1006844},
        'raised': {6962, 58512, 60516, 61103, 80469, 85491, 94437, 94524, 116556, 125124, 165233, 185106, 222793, 231913, 234829, 238321, 244816, 253130, 264975, 275946, 275958, 282919, 293993, 294043, 302357, 305192, 308649, 315404, 316111, 320636, 392341, 429254, 431867, 431870, 432207, 444736, 447251, 486993, 488383, 510305, 511770, 515521, 537062, 537275, 543490, 566084, 568212, 570926, 574094, 575914, 576594, 580506, 583107, 586520, 701097, 703253, 735409, 748760, 754485, 757556, 757604, 768041, 776067, 791019, 831144, 853508, 884373, 888685, 899910, 903512, 903602, 939832, 952561, 956100, 965107, 968079, 974788, 975073, 983868, 1010585, 1011033},
        'wordart': {31715, 39716, 44876, 74919, 75133, 104696, 108556, 110006, 113592, 117784, 140866, 143122, 143125, 145951, 149959, 150049, 150213, 150279, 150281, 150428, 150490, 151687, 154129, 156874, 159778, 159895, 160120, 160247, 160276, 160283, 162846, 163105, 163216, 164079, 164761, 166267, 168230, 171234, 171790, 172286, 172298, 172308, 172335, 175266, 175334, 175831, 176590, 176806, 177410, 177554, 179221, 179305, 180561, 181293, 181310, 182794, 183106, 183209, 183290, 184192, 185667, 186031, 186403, 186514, 187030, 187343, 190446, 194951, 197172, 198183, 198237, 200187, 201238, 201769, 202088, 202370, 202382, 203726, 207168, 207507, 208267, 208991, 212503, 212525, 213482, 223470, 226107, 227287, 232970, 233894, 233940, 234845, 236981, 237720, 241485, 242131, 244615, 244620, 244624, 245553, 246792, 247779, 247807, 251942, 254875, 255095, 258723, 259914, 259966, 262937, 263127, 263146, 263239, 267506, 268151, 268235, 276946, 278369, 283787, 290607, 292262, 292299, 292365, 293734, 294791, 296802, 297370, 298140, 298342, 305806, 306567, 306734, 307309, 310559, 310642, 312263, 312802, 327995, 329859, 335910, 337662, 338453, 343304, 353405, 353413, 353893, 354529, 355161, 355494, 355623, 355691, 355774, 355950, 355954, 356051, 356179, 356242, 356497, 356706, 357194, 357264, 373665, 373717, 376825, 404495, 419551, 422019, 422612, 427949, 433690, 435115, 435506, 436128, 436294, 436315, 436376, 436654, 437268, 437576, 437642, 438235, 438558, 439054, 439080, 439472, 439674, 441572, 455623, 476376, 481154, 485066, 527590, 529010, 531947, 538763, 539110, 540596, 542057, 542098, 545024, 546352, 551044, 556749, 560492, 560528, 565064, 567446, 585918, 611633, 675438, 676105, 678361, 678431, 682537, 683559, 683671, 684947, 714507, 747645, 748163, 749913, 764697, 770430, 772118, 772585, 774869, 774876, 776310, 777338, 796911, 798872, 817377, 826692, 833591, 838845, 840866, 876796, 878192, 883355, 887083, 887166, 887344, 894624, 903599, 912879, 915091, 922486, 935850, 945271, 945286, 946310, 947397, 951044, 951139, 951591, 952854, 952928, 953157, 954346, 959683, 960200, 962644, 968523, 977304, 981412, 988359, 990304, 998700, 1000188, 1003561},
    }
    for attr, imgset in sorted(belongs.items()):
        random.shuffle(imgset)
        for id in forbidden[attr]:
            imgset.remove(id)
        imgset = list(filter(lambda id: min(all[id][0].shape[:2]) > 16, imgset))
        if 'distorted' in attr:
            imgset = list(filter(lambda id: polygon_in_center(polygons[id], 512), imgset))
        else:
            imgset = list(filter(lambda id: polygon_in_center(polygons[id], 10), imgset))
        imgset = imgset[:x * y]
        if not os.path.isdir(settings.ATTR_SAMPLE_DIR):
            os.makedirs(settings.ATTR_SAMPLE_DIR)
        file_path = os.path.join(settings.ATTR_SAMPLE_DIR, '{}.png'.format(attr))
        print(file_path)
        canvas = np.zeros((y * cropsize, x * cropsize, 3), dtype=np.uint8)
        for i, j in enumerate(imgset):
            cropped = all[j][0]
            resized = cv2.resize(cropped, (cropsize, cropsize))
            canvas[(i // x) * cropsize:(i // x + 1) * cropsize, (i % x) * cropsize:(i % x + 1) * cropsize] = resized
        cv2.imwrite(file_path, canvas)
        with open(os.path.join(settings.ATTR_SAMPLE_DIR, '{}.json'.format(attr)), 'w') as f:
            json.dump(imgset, f)


def crop_no_concat():
    assert six.PY3
    random.seed(0)

    belongs = defaultdict(list)
    with open(settings.TRAIN) as f:
        lines = f.read().splitlines()
    with open(settings.VAL) as f:
        lines += f.read().splitlines()
    for line in lines:
        anno = json.loads(line.strip())
        i = 0
        for char in anno_tools.each_char(anno):
            if not char['is_chinese']:
                continue
            for attr in settings.ATTRIBUTES:
                if attr in char['attributes']:
                    belongs[attr].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
                else:
                    belongs['not-{}'.format(attr)].append((anno['image_id'], anno['file_name'], char['adjusted_bbox'], i))
            i += 1

    for attr, imgset in sorted(belongs.items()):
        imgset = random.sample(imgset, 200)
        root = os.path.join(settings.ATTR_SAMPLE_DIR, attr)
        resized_root = os.path.join(settings.ATTR_SAMPLE_DIR, '{}-resized'.format(attr))
        if not os.path.isdir(root):
            os.makedirs(root)
        if not os.path.isdir(resized_root):
            os.makedirs(resized_root)
        for i, (image_id, file_name, bbox, idx_in_img) in enumerate(imgset):
            if i % 100 == 0:
                print(attr, i, '/', len(imgset))
            image = cv2.imread(os.path.join(settings.TRAINVAL_IMAGE_DIR, file_name))
            cropped_file_path = os.path.join(root, '{}_{}.png'.format(image_id, idx_in_img))
            resized_file_path = os.path.join(resized_root, '{}_{}.png'.format(image_id, idx_in_img))
            cropped = predictions2html.crop(image, bbox, float('inf'))
            resized = cv2.resize(cropped, (32, 32))
            cv2.imwrite(cropped_file_path, cropped)
            cv2.imwrite(resized_file_path, resized)


if __name__ == '__main__':
    main()
