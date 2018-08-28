#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Implementation of the Machine Comprehension model
@Ref      : GitHub:Baidu/DuReader
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : model.py
@Time     : 18-8-28 下午2:33
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize


class ReadingComprehension(object):
    """

    """