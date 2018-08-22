#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
@Ref      : https://www.kesci.com/home/project/5b2ca2e3f110337467b2752c
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : DuReader 
@File     : get_train_label.py
@Time     : 18-8-21 下午2:24
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import numpy as np
import string


def find_first_meaningful_token(tokens):
    """
    找到第一个有意义的符号
        有：i
        无：None
    :param tokens:
    :return:
    """
    for i, token in enumerate(tokens):
        if len(token) <= 1 and not (
                u'\u4e00' <= token <= u'\u9fff' or token in string.ascii_letters or token in string.digits):
            # \u4e00 - \u9fff 是汉字字符集
            # string.ascii_letters是生成所有字母，从a-z和A-Z
            # string.digits是生成所有数字0-9
            # 符号长度<=1;不在(汉字字符表; 字母表; 数字表)中
            continue
        else:
            return i


def segment_match_p_a(p_tokens, a_tokens):
    """
    得到匹配区间
        format: p_start, p_end, len, a_end, mismatch
    :param p_tokens:
    :param a_tokens:
    :return:
    """
    match = []
    # match from every start point
    for p_start, p_token in enumerate(p_tokens):
        if p_token == a_tokens[0]:
            # 找到与a第一个符号（元素）一样的
            m = p_start + 1
            n = 1
            mismatch = 0
            while True:
                try:
                    if p_tokens[m] == a_tokens[n]:
                        m += 1
                        n += 1
                        continue
                    elif p_tokens[m] == a_tokens[n + 1]:
                        m += 1
                        n += 2
                        mismatch += 1
                    elif p_tokens[m + 1] == a_tokens[n]:
                        m += 2
                        n += 1
                        mismatch += 1
                    elif p_tokens[m + 1] == a_tokens[n + 1]:
                        m += 2
                        n += 2
                        mismatch += 2
                    elif p_tokens[m] == a_tokens[n + 2]:
                        m += 1
                        n += 3
                        mismatch += 2
                    elif p_tokens[m + 2] == a_tokens[n]:
                        m += 3
                        n += 1
                        mismatch += 2
                    elif p_tokens[m + 2] == a_tokens[n + 1]:
                        m += 3
                        n += 2
                        mismatch += 3
                    elif p_tokens[m + 1] == a_tokens[n + 2]:
                        m += 2
                        n += 3
                        mismatch += 3
                    elif p_tokens[m + 2] == a_tokens[n + 2]:
                        m += 3
                        n += 3
                        mismatch += 4
                    else:
                        break
                except:
                    break
            mismatch -= len(list(filter(lambda _: _ in '。，：；、.,;:', a_tokens)))
            match.append((p_start, m - 1, m - p_start, n - 1, mismatch))  # p_start, p_end, len, a_end, mismatch
    if not match:
        return None
    match = np.array(match)
    return match[match[:, 2].argmax()]


def match_p_a(p_tokens, a_tokens):
    """
    匹配篇章和答案
        返回(p_start, p_end, len)
    :param p_tokens:
    :param a_tokens:
    :return:
    """
    matches = []
    # 开始搜索的位置
    p_end = 0
    a_end = 0
