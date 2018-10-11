#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : search.py
@Time     : 18-9-30 下午5:55
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from ir.config import Config
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Search(object):
    def __init__(self):
        print("Searching...")
        self.logger_search = logging.getLogger('searching')

    def search_by_question(self, question, top_n, config):
        """
        Search candidate paragraphs
        :param question:
        :param top_n:
        :param config:
        :return:
        """
        q = {
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["title^2", "paragraph"],
                    "fuzziness": "AUTO"
                }
            },
            # "_source": ["title", "paragraph"],
            # "size": 3
        }

        count = 0
        while count < 5:
            try:
                res = config.es.search(index=config.index_name, doc_type=config.doc_type, body=q, request_timeout=30)
                topn = res['hits']['hits']
                count = 0
                result = []
                for data in topn:
                    if count < top_n:
                        result.append((data['_source']['title'], data['_source']['paragraph']))
                        count += 1
                return result
            except:
                self.logger_search.info("Try again!")
                count += 1
                continue

        self.logger_search.info("ReadTimeOutError may not be covered!")
        return []


def main():
    config = Config()
    search = Search()
    query = "围棋多少颗子？"
    result = search.search_by_question(query, 1, config)
    for data in result:
        print(data[0], data[1])
    print(result[0][1])


if __name__ == '__main__':
    main()
