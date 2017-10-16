# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def each_char(image_anno):
    for block in image_anno['annotations']:
        for char in block:
            yield char
