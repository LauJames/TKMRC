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
        print(str(p_start) + ': ' + p_token)
        if p_token == a_tokens[0]:
            # 找到与a第一个符号（元素）一样的
            m = p_start + 1
            n = 1
            # 记录这一段paragraph的tokens的mismatch分数
            mismatch = 0
            while True:
                # print(p_token)
                print(mismatch)
                try:
                    print('p_tokens:' + p_tokens[m])
                    print('a_tokens:' + a_tokens[n])
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
                    # start: +3 意味着后面还可以继续扩展，这里只计算到这里，避开一般的标点符号就行，按照1/2/3的步长进行搜索
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
                    # end: +3 意味着后面还可以继续扩展，这里只计算到这里，避开一般的标点符号就行，按照1/2/3的步长进行搜索
                    else:
                        break
                except IndexError as e:
                    print(e)
                    break
            # 减去标点符号导致的mismatch
            mismatch -= len(list(filter(lambda _: _ in '。，：；、.,;:', a_tokens)))
            match.append((p_start, m - 1, m - p_start, n - 1, mismatch))
            #  候选匹配区间信息
            # (p_start, p_end, len, a_end, mismatch)
    if not match:
        return None
    match = np.array(match)
    return match[match[:, 2].argmax()]  # 取匹配最长的paragraph的tokens作为匹配区间


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
    while True:
        if a_end >= len(a_tokens) or p_end >= len(p_tokens):
            break
        # 从零开始搜索匹配区间
        match = segment_match_p_a(p_tokens[p_end:], a_tokens[a_end:])
        # start ************* 应对当前位置无匹配区间
        if match is None:
            # 从下一个开始，否则死循环指示当前位置
            a_s = find_first_meaningful_token(a_tokens[a_end + 1:])
            if a_s is None:
                break
            # 找到第一个有意义的token之后，标记为answer start
            a_end += (a_s + 1)
            # 然后+1标记为answer end
            continue
        # end ************* 应对当前位置无匹配区间

        p_s, p_e, _, a_e, _ = match
        matches.append([p_end + p_s, p_end + p_e, p_e - p_s + 1])

        # 换算坐标系
        a_end += (a_e + 1)
        p_end += (p_e + 1)

        # 下一个开始的token
        a_s = find_first_meaningful_token(a_tokens[a_end:])
        if a_s is None:
            # time to the end
            break
        a_end += a_s

    return matches


def merge_spans(spans):
    """
    合并span
    :param spans:
    :return:
    """
    # 严格标准, 单个字符匹配的情况，不考虑
    if spans:
        new_spans = [spans[0]]
        for i in range(1, len(spans)):
            if sum([_[2] for _ in spans]) < 20:
                threshold = 6
            else:
                threshold = 10
            if spans[i][0] - new_spans[-1][1] <= threshold:
                new_spans[-1][1] = spans[i][1]
                new_spans[-1][2] += spans[i][2]  # 长度
            else:
                new_spans.append(spans[i])
        return new_spans
    else:
        return []


def match_all_p_a(tokenized_paras, tokenized_answers, max_a_num=5):
    """
    匹配所有的篇章和答案
    :param tokenized_paras:
    :param tokenized_answers:
    :param max_a_num:
    :return:
    """
    # print(tokenized_paras)
    if not tokenized_answers:
        return []
    p_a_list = []
    for p_idx, p_tokens in enumerate(tokenized_paras):
        candidate_spans = []
        for a_idx, a_tokens in enumerate(tokenized_answers):
            if not a_tokens:
                continue
            spans = merge_spans(match_p_a(p_tokens, a_tokens))
            if not spans:
                continue
            # 筛选
            # 忽略末尾无意义的字符
            len_a_token = len(a_tokens)
            if not find_first_meaningful_token(a_tokens[-1]):
                len_a_token -= 1

            spans = np.array(spans)
            max_match_span = spans[spans[:, 2].argmax()]
            # print(spans)
            print('******************')
            # 分情况讨论
            # 1. 答案本身比较短，则必须是单个答案，并且接近完全匹配
            if len(a_tokens) <= 3:
                if max_match_span[2] >= len_a_token:
                    candidate_spans.append((max_match_span, -1))
            # 2. 答案比较长，那么就可以分段
            else:
                # 2.1 第一个就直接很高，就不用添加模型复杂度
                if max_match_span[2] > 0.8 * len_a_token:
                    candidate_spans.append((max_match_span, 2))
                # 2.2 都是小片段
                else:
                    spans = spans[spans[:, 2].argmax()]
                    max_match_len = spans[2]
                    if max_match_len > 0.3 * len_a_token:  # 由于截取的问题，很多都是一部分，没办法
                        candidate_spans.append((spans, max_match_span[2] / len(a_tokens)))

        if candidate_spans:
            for candidate_span in candidate_spans:
                if candidate_span[1] == -1:
                    # 存在很短的情况
                    p_a_list.append((p_idx, candidate_span[0], 1))
                    break
            else:
                target_span = sorted(candidate_spans, key=lambda i: i[1], reverse=True)[0]
                target_span = [p_idx] + list(target_span)
                p_a_list.append(target_span)
        else:
            p_a_list.append((p_idx, None, -2))

    return p_a_list


def get_label(tokenized_paras, tokenized_answer, max_a_num=5):
    """
        得到最终的span
    :param tokenized_paras:
    :param tokenized_answer:
    :param max_a_num:
    :return:
    """
    matches = match_all_p_a(tokenized_paras, tokenized_answer, max_a_num)
    if not matches:
        return -1, -1, -1
    matches = sorted(matches, key=lambda i: i[2], reverse=True)
    match = matches[0]
    if match[2] == -2:
        return -1, -1, -1
    answer_doc = match[0]
    answer_start = match[1][0]
    answer_end = match[1][1]
    return answer_doc, answer_start, answer_end
