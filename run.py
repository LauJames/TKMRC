#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
The prepare for model data and running model
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : run.py
@Time     : 18-8-26 下午4:31
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import pickle
import argparse
import logging
from data.data_loader import BRCDataset
from data.vocab_glove import Vocab


def parse_args():
    """
    Parse command line arguments.
    :return:
    """
    parser = argparse.ArgumentParser('Reading Comprehension ')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answer for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optimizer', default='adam',
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
    model_settings.add_argument('--algorithm', choices=['R-net', 'QA-net', 'BIDAF'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_dim', type=int, default=300,
                                help='dimension of the embeddings')
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

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/DuReaderDemo/search.train.json'],
                               help='list of files that contain the preprocessed train set')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/DuReaderDemo/search.dev.json'],
                               help='list of files that contain the preprocessed deviation set')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/DuReaderDemo/search.test.json'],
                               help='list of files that contain the preprocessed test set')
    path_settings.add_argument('--brc_dir', default='./data/DuReaderDemo',
                               help='the directory with preprocessed xx reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/',
                               help='the directory to save vocabulary')
    path_settings.add_argument('--model_dir', default='./models/',
                               help='the directory to save models')
    path_settings.add_argument('--result_dir', default='./results/',
                               help='the dir to save the final output')
    path_settings.add_argument('--summary_dir', default='./summary/',
                               help='the directory to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    return parser.parse_args()

