#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Generate vocab from Glove Vectors.txt
@Ref      : https://www.kesci.com/home/project/5b2ca2e3f110337467b2752c
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : DuReader 
@File     : vocab_glove.py
@Time     : 18-8-20 下午8:07
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import numpy as np
import logging
import pickle as pkl
from sklearn.externals import joblib


class Vocab(object):
    def __init__(self, lower=True):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.vocab_size = None
        self.embedding = None

    def prepare(self, filename):
        """
        准备符号和id对应的字典
        :param filename:
        :return:
        """
        tokens = []
        embs = []
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.readlines()
            for line in content:
                line = line.split(' ')  # 以空格分割，否则某些字符会干扰分割
                tokens.append(line[0])
                print('tokens:' + str(line[0]))
                embs.append(line[1])
                print('embs dims:' + str(len(line[1:])))  # 也可以指定1：201
                print('embs:' + str(line[1]))

        ids = list(range(len(tokens)))
        self.token2id = dict(zip(tokens, ids))
        self.id2token = dict(zip(ids, tokens))

        self.embedding = np.array(embs, dtype=np.float32)
        self.vocab_size = self.embedding.shape[0]

    def get_id(self, token):
        """
        得到一个符号的id
        :param token:
        :return:
        """
        token = token.encode('utf-8')
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id['<unk>']

    def convert2ids(self, tokens):
        """
        符号列表转为id列表
        :param tokens:
        :return:
        """
        vec = [self.get_id(label) for label in tokens]
        return vec


def main():
    """
    test
    :return:
    """
    logger = logging.getLogger('prepare')
    logger.info('test the Building vocabulary...')
    vocab = Vocab(lower=True)
    vocab.prepare('./temp/vectors.txt')

    logger.info('Saving vocab...')

    # test_dump = open('./temp/test.pkl', 'wb')
    # pkl.dump(vocab, test_dump)
    # test_dump.close()

    joblib.dump(vocab, './temp/test.pkl')

    logger.info('Test done')


if __name__ == '__main__':
    main()
