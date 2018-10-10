#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : config.py
@Time     : 18-9-30 下午5:54
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from elasticsearch import Elasticsearch


class Config(object):
    def __init__(self):
        print("config...")
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.index_name = "mrc"
        self.doc_type = "paragraph"
        # test
        # file_path = "../data/DuReaderDemo/search.dev.json"
        file_path = '../data/mergeData/merge.all.json'
        self.doc_path = file_path


def main():
    Config()


if __name__ == '__main__':
    main()
