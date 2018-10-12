#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Generate the corpus to train word embedding
@Ref      : https://www.kesci.com/home/project/5b2ca2e3f110337467b2752c
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : DuReader 
@File     : gen_corpus.py
@Time     : 18-8-20 下午2:36
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import json
from pathlib import Path
# 所有段落数据都应纳入
data_home_path = Path('./merge_processed/')
data_home_path_tk = Path('./TKData/')


def gen_corpus(data_path):
    """
    generate corpus(lines of blank split words)
    :param data_path:
    :return:
    """
    x = []
    for i in data_path.rglob('*.json'):
        # Path.rglob():递归遍历所有子目录的文件---支持后缀匹配
        print(i)
        with i.open() as f:
            for line in f:
                item = json.loads(line)
                for doc in item['documents']:
                    for para in doc['segmented_paragraphs']:
                        x.append(' '.join(para))
                        x += '\n'
    x = '\n'.join(x)
    print(x)
    with open('./temp/MRCcorpus', 'w+') as f:
        f.write(x)


def tk_gen_corpus(data_path):
    """
    generate corpus for tk machine reading comprehension dataset
    :param data_path:
    :return:
    """
    x = []
    for i in data_path.rglob('*.json'):
        # Path.rglob():递归遍历所有子目录的文件---支持后缀匹配
        print(i)
        with i.open() as f:
            for line in f:
                item = json.loads(line)
                for doc in item['documents']:
                    for para in doc['segmented_paragraphs']:
                        x.append(' '.join(para))
                        x += '\n'
    x = '\n'.join(x)
    print(x)
    with open('./temp/TKMRCCorpus', 'w+') as f:
        f.write(x)


if __name__ == '__main__':
    # tk_gen_corpus(data_home_path)
    gen_corpus(data_home_path)
