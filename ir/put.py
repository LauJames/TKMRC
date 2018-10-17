#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : put.py
@Time     : 18-10-17 下午4:54
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from ir.config import Config
from elasticsearch import helpers

import pandas as pd
import json
import jieba
import time

excel_path = './doc/tk-QP.xlsx'
json_path = './doc/excel2json.temp.json'


def excel2json():
    """
    input: excel files:	问题	段落 all of them are string type
    transform the raw data to json type including title and paragraph
    :return:
    """
    dataframe = pd.read_excel(excel_path,
                              sheet_name='Sheet1',
                              header=0,
                              dtype={'问题': str, '段落': str})
    dataframe.dropna()

    titles = []
    paragraphs = []

    for key, data in dataframe.iterrows():
        titles.append(data['问题'])
        paragraphs.append(data['段落'])

    data_list = []
    for title, paragraph in zip(titles, paragraphs):
        paragraph = paragraph.replace('\n', '')
        paras_data = {"paragraph": paragraph, "title": title}
        data_list.append(paras_data)

    with open(json_path, 'w', encoding='utf-8') as temp_json:
        for json_line in data_list:
            data_json = json.dumps(json_line, ensure_ascii=False)
            temp_json.write(str(data_json))
            temp_json.write('\n')


def get_json_obj(json_path):
    """
    Read json file and load to dict list
    :param json_path:
    :return: dict list
    """
    paras = {}
    t_id = int(time.time())  # according time to set id
    with open(json_path, 'r') as fin:
        line = fin.readline()
        while line:
            line = json.loads(line.strip(), encoding='utf-8')

            paragraph = ' '.join(token for token in jieba.cut(line['paragraph'].strip()))
            title = ' '.join(token for token in jieba.cut(line['title'].strip()))

            t_id += 1
            paras[t_id] = {'title': title, 'paragraph': paragraph}

            line = fin.readline()
    return paras


def put2es(paras, bulk_size, config):
    """
    Put paras into es
    :param paras:
    :param bulk_size:
    :param config:
    :return:
    """
    count = 1
    actions = []
    for para_id, para in paras.items():
        action = {
            "_index": config.index_name,
            "_type": config.doc_type,
            "_id": para_id,
            "_source": para
        }

        actions.append(action)
        count += 1

        if len(actions) % bulk_size == 0:
            helpers.bulk(config.es, actions)
            print("bulk index:" + str(count))
            actions = []

    if len(actions) > 0:
        helpers.bulk(config.es, actions)
        print("bulk index:" + str(count))


if __name__ == '__main__':

    config = Config()

    # excel2json()
    title_paras = get_json_obj(json_path)
    # for idx, title_para in title_paras.items():
    #     print(idx)
    #     print(title_para)

    put2es(title_paras, bulk_size=10000, config=config)
