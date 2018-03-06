# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import imp
import settings

from pythonapi import common_tools


def main():
    if not common_tools.exists_and_newer(settings.TEST_LIST, settings.CATES):
        dn_prepare = imp.load_source('dn_prepare', '../detection/prepare_test_data.py')
        dn_prepare.crop_test_images(settings.TEST_LIST)


if __name__ == '__main__':
    main()
