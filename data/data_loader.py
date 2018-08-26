#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
@Ref1     : https://github.com/LauJames/DuReader/blob/master/tensorflow/dataset.py
@Ref2     : https://www.kesci.com/home/project/5b2ca2e3f110337467b2752c
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : data_loader.py
@Time     : 18-8-22 下午5:12
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import json
import logging

import numpy as np

from data.get_train_label import get_label
from data.vocab_glove import Vocab
import pickle


class BRCDataset(object):
    def __init__(self, max_p_num, max_p_len, max_q_len, vocab,
                 train_files=[], dev_files=[], test_files=[],
                 train_batch_size=32, infer_batch_size=32,
                 max_samples=1000000, skip_samples=1):
        self.logger = logging.getLogger("BRC")
        self.vocab = vocab

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size

        self.max_p_num = max_p_num
        self.max_a_num = 5
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.train_set, self.dev_set, self.test_set = [], [], []

        cols = ['p0', 'p1', 'p2', 'p3', 'p4', 'q', 's', 'e', 'meta']
        self.feeding = dict(zip(cols, range(len(cols))))

        if train_files:
            for train_file in train_files:
                self.train_set += self.load_dataset(train_file)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self.load_dataset(dev_file)
            self.logger.info('Dev set size {} questions.'.format(len(self.dev_set)))
        if test_files:
            for test_file in test_files:
                self.test_set += self.load_dataset(test_file)
            self.logger.info('Test set size {} questions.'.format(len(self.test_set)))

    def load_dataset(self, data_path):
        """
        加载数据集
        :param data_path:
        :return:
        """
        with open(data_path, 'r') as f:
            data_set = []
            for line_idx, line in enumerate(f):
                # print(f)
                if line_idx > self.max_samples:
                    self.logger.info('Sample num reached the upper bound.')
                    break
                if line_idx % 10000 == 0:
                    self.logger.info('Already read {} samples.'.format(line_idx))
                if line_idx % self.skip_samples != self.skip_samples - 1:
                    continue
                sample = json.loads(line.strip())
                sample['question_tokens'] = sample['segmented_question']
                sample['passage_list_tokens'] = []  # list of list
                for d_idx, doc in enumerate(sample['documents']):
                    if d_idx >= self.max_p_num:
                        break
                    para = []
                    for p in doc['segmented_paragraphs']:
                        para += p
                        para.append('\n')
                    # 减少临时变量消耗
                    # for p in doc['segmented_paragraphs']:
                    #     para.append(p)
                    # para = '\n'.join(para)
                    if len(para) != 0:
                        sample['passage_list_tokens'].append(para[:self.max_p_len])
                if len(sample['passage_list_tokens']) != 0:
                    for i in range(5 - len(sample['passage_list_tokens'])):
                        sample['passage_list_tokens'].append([])
                    if 'answers' in sample:
                        answer_doc, answer_start, answer_end = get_label(sample['passage_list_tokens'],
                                                                         sample['segmented_answers'],
                                                                         self.max_a_num)
                        if answer_doc == -1:
                            continue
                        sample['answer_doc'] = answer_doc
                        sample['start_id'] = answer_start
                        sample['end_id'] = answer_end
                        del sample['segmented_an已经存在swers']
                        del sample['fake_answers']
                        del sample['answer_docs']
                        del sample['answer_spans']
                        del sample['match_scores']
                        del sample['question']
                    else:
                        sample['start_id'] = 0
                        sample['end_id'] = 0
                    del sample['documents']
                    del sample['segmented_question']
                    del sample['fact_or_opinion']
                    data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        生成一个batch
        :param data:
        :param indices:
        :return:
        """
        batch_data = []
        for i in indices:
            sample = data[i]
            start_labels = []
            end_labels = []
            passage_list_tokens_ids = []
            question_token_ids = self.vocab.convert2ids(sample['question_tokens'])
            for j, _ in enumerate(sample['passage_list_tokens']):
                passage_tokens_ids = self.vocab.convert2ids(_)
                if passage_tokens_ids:
                    passage_list_tokens_ids.append(passage_tokens_ids)
                    start_label = [[0.0]] * len(_)
                    end_label = [[0.0]] * len(_)
                else:
                    passage_list_tokens_ids.append([0])
                    start_label = [[0.0]]
                    end_label = [[0.0]]

                if 'answer_doc' in sample and sample['answer_doc'] == j:
                    start_label[sample['start_id']] = [1.0]
                    end_label[sample['end_id']] = [1.0]
                start_labels += start_label
                end_labels += end_label
            meta = {'question_id': sample['question_id'],
                    'passage_list': sample['passage_list'],
                    'question_type': sample['question_type']}
            if 'answers' in sample:
                meta['answers'] = sample['answers']
            batch_data.append(passage_list_tokens_ids + [question_token_ids, start_labels, end_labels, meta])

        return batch_data

    def _gen_mini_batches(self, data, batch_size, shuffle=True):
        """
        shuffle 数据集，产生batch
        :param data:
        :param batch_size:
        :param shuffle:
        :return:
        """
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            # yield self._one_mini_batch(data, batch_indices)
            return self._one_mini_batch(data, batch_indices)


def main():
    """
    test the data loader
    :return:
    """
    logger = logging.getLogger("BRC")
    logger.info('Building vocabulary ...')
    vocab = Vocab(lower=True)
    vocab.prepare('./temp/vocab.txt')

    logger.info('Saving vocab ...')

    pkl_in = open('./temp/vocab.pkl', 'wb')
    pickle.dump(vocab, pkl_in)

    pkl_out = open('./temp/vocab.pkl', 'rb')
    pickle.load(pkl_out)
    train_batch_size = 32
    infer_batch_size = 32

    max_p_num = 5
    max_p_len = 500
    max_q_len = 60
    max_samples = 100
    skip_samples = 1
    train_files = ['./DuReaderDemo/search.train.json']
    dev_files = ['./DuReaderDemo/search.dev.json']
    test_files = ['./DuReaderDemo/search.test.json']

    brc_test_data = BRCDataset(
        max_p_num=max_p_num,
        max_p_len=max_p_len,
        max_q_len=max_q_len,
        vocab=vocab,
        # train_files=train_files,
        dev_files=dev_files,
        # test_files=test_files,
        train_batch_size=train_batch_size,
        infer_batch_size=infer_batch_size,
        max_samples=max_samples,
        skip_samples=skip_samples
    )
    print(brc_test_data)


if __name__ == '__main__':
    main()
