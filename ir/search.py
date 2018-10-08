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


class Search(object):
    def __init__(self):
        print("Searching...")
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def search_by_question(self, question, top_n, config):
        """
        Search candidate paragraphs
        :param question:
        :param top_n:
        :param config:
        :return:
        """
        q = {"query":
                 {"bool":
                      {"must":
                           {"match":
                                {"paragraph": question}
                            }
                       }
                  }
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
                        result.append(data['_source']['paragraph'])
                        count += 1
                return result
            except:
                logging.info("Try again!")
                count += 1
                continue

        logging.info("ReadTimeOutError may not be covered!")
        return []


def main():
    config = Config()
    search = Search()
    query = "小说排行榜第一位是哪本小说？"
    result = search.search_by_question(query, 3, config)
    for data in result:
        print(data)


if __name__ == '__main__':
    main()
