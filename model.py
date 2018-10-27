#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Implementation of the Machine Comprehension model
@Ref      : GitHub:Baidu/DuReader
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : model.py
@Time     : 18-8-28 下午2:33
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize


class BiDAF(object):
    """
    Implements the Bi-Directional Attention Flow Reading Comprehension Model
    """
    def __init__(self, vocab, args):
        # logging
        self.logger = logging.getLogger("BiDAF")

        self.hidden_size = args.hidden_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # vocab
        self.vocab = vocab

        # build the computation graph with tensorflow
        start_time = time.time()

        # Placeholders for input, output and dropout
        self.p = tf.placeholder(tf.int32, [None, None], name='input_p')
        self.q = tf.placeholder(tf.int32, [None, None], name='input_q')
        self.p_length = tf.placeholder(tf.int32, [None], name='p_len')
        self.q_length = tf.placeholder(tf.int32, [None], name='q_len')
        self.start_label = tf.placeholder(tf.int32, [None], name='start')
        self.end_label = tf.placeholder(tf.int32, [None], name='end')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # embedding layer
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True,
                name='word_embeddings'
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

        # Encodding layer

