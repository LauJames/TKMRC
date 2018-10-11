#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : inference.py
@Time     : 18-9-30 下午5:02
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


from ir.config import Config
from ir.search import Search
from tf_version.dataset import BRCDataset
from tf_version.vocab import Vocab
from tf_version.rc_model import RCModel

import logging
import jieba
import pickle
import argparse
import os
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename='./test_log/test.log')


def path_arg4test():
    """
    Relevant path settings for test program
    :return: 
    """
    parser = argparse.ArgumentParser('Reading Comprehension test demo')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_setting = parser.add_argument_group('path settings')
    path_setting.add_argument('--vocab_dir',  default='../data/vocab/',
                              help='the dir to save vocabulary')
    path_setting.add_argument('--model_dir', default='../data/models/',
                              help='the dir to store models')
    path_setting.add_argument('--log_path', default='./test_log/test.log',
                              help='path of the log file. If not set, logs are print to console ')
    return parser.parse_args()


if __name__ == '__main__':
    ir_config = Config()
    search = Search()
    args = path_arg4test()

    logger = logging.getLogger('test')

    # load vocab
    logger.info('Loading vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    logger.info('Demo Testing...')
    test_query = "围棋有多少颗棋子?"
    result = search.search_by_question(test_query, 3, ir_config)
    passage_para = result[0][1]
    test_query_seg = [token for token in jieba.cut(test_query)]
    paragraph_seg = [token for token in jieba.cut(passage_para)]
    query_ids = [vocab.get_id(label) for label in paragraph_seg]
    para_ids = [vocab.get_id(label) for label in test_query_seg]
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, demo_string=[test_query_seg, paragraph_seg])

    # padding
    pad_id = vocab.get_id(vocab.pad_token)
    pad_p_len = min(args.max_p_len, len(paragraph_seg))
    pad_q_len = min(args.max_q_len, len(test_query_seg))
    query_ids = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len] for ids in query_ids]
    para_ids = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len] for ids in para_ids]

    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)

    feed_dict = {
        rc_model.p: query_ids,
        rc_model.q: para_ids,
        rc_model.p_length: pad_p_len,
        rc_model.q_length: pad_q_len,
        rc_model.start_label: [0],
        rc_model.end_label: [0],
        rc_model.dropout_keep_prob: 1.0}
    start_probs, end_probs = rc_model.sess.run([rc_model.start_probs, rc_model.end_probs], feed_dict)
