#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : index.py
@Time     : 18-9-30 下午5:54
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


from ir.config import Config
from elasticsearch import helpers
import jieba
import logging
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Index(object):
    def __init__(self):
        print("Indexing...")
        self.logger_index = logging.getLogger('indexing')

    def data_convert(self, file_path="../data/DuReaderDemo/search.dev.json"):
        """
        Data convert program for Baidu's DuReader raw json data
        :param file_path:
        :return:
        """
        self.logger_index.info("convert raw json data into single doc")
        paras = {}
        para_id = 0
        with open(file_path, 'r') as f:
            line = f.readline()
            while line:
                line = json.loads(line.strip(), encoding='utf-8')

                for document in line["documents"]:
                    # question_id = line["question_id"]
                    # question_type = line["question_type"]
                    # segmented_question = ' '.join(token for token in line["segmented_question"])
                    # paras = document["segmented_paragraphs"]
                    para_list = document["paragraphs"]
                    title = ' '.join(token for token in jieba.cut(document["title"].strip()))
                    for para in para_list:
                        paragraph = ' '.join(token for token in jieba.cut(para.strip()))
                        print('title: ' + title)
                        print('paragraph: ' + paragraph)
                        paras[para_id] = {'title': title, 'paragraph': paragraph}
                        para_id += 1
                line = f.readline()
        self.logger_index.info(str(para_id) + 'title and paragraphs loaded!')
        return paras

    def create_index(self, config):
        """
        Creating index for paragraphs
        :param config:
        :return:
        """
        self.logger_index.info("creating '%s' index..." % config.index_name)
        request_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {
                    "LM": {
                        "type": "LMJelinekMercer",
                        "lambda": 0.4
                    }
                }
            },
            "mappings": {
                config.index_name: {
                    "properties": {
                        "title": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            # 支持参数yes（term存储），
                            # with_positions（term + 位置）,
                            # with_offsets（term + 偏移量），
                            # with_positions_offsets(term + 位置 + 偏移量)
                            # 对快速高亮fast vector highlighter能提升性能，但开启又会加大索引体积，不适合大数据量用
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        },
                        "paragraph": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        }
                    }
                }
            }
        }
        # 删除先前的索引
        config.es.indices.delete(index=config.index_name, ignore=[400, 404])
        res = config.es.indices.create(index=config.index_name, body=request_body)
        self.logger_index.info(res)
        self.logger_index.info("Indices are created successfully")

    def bulk_index(self, paras, bulk_size, config):
        """
        Bulk indexing paras
        :param paras:
        :param bulk_size:
        :param config:
        :return:
        """
        self.logger_index.info("Bulk index for paragraphs")
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
                self.logger_index.info("bulk index: " + str(count))
                actions = []

        if len(actions) > 0:
            helpers.bulk(config.es, actions)
            self.logger_index.info("bulk index: " + str(count))


def main():
    config = Config()
    index = Index()
    paras = index.data_convert(config.doc_path)
    index.create_index(config)
    index.bulk_index(paras=paras, bulk_size=10000, config=config)


if __name__ == '__main__':
    main()
