#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class Seq2Seq(chainer.Chain):

    # test ok
    def __init__(self, src_vocab_size, src_embed_size, hidden_size, dst_embed_size, dst_vocab_size):
        """
        :param src_vocab_size: the number of source vocabulary 
        :param src_embed_size: the number of source embedded units
        :param hidden_size: the number of hidden units 
        :param dst_embed_size: the number of destination embedded units
        :param dst_vocab_size: the number of destination vocabulary units
        """
        super(Seq2Seq, self).__init__(
            w_xi=L.EmbedID(src_vocab_size, src_embed_size),
            w_ip=L.LSTM(src_embed_size, hidden_size),
            w_pq=L.LSTM(hidden_size, hidden_size),
            w_qj=L.Linear(hidden_size, dst_embed_size),
            w_jy=L.Linear(dst_embed_size, dst_vocab_size),
            w_yq=L.LSTM(dst_vocab_size, hidden_size)
        )

        self.train = True

    # test ok
    def reset_state(self):
        self.w_ip.reset_state()
        self.w_pq.reset_state()
        self.w_yq.reset_state()

    def encode(self, x):
        i = F.tanh(self.w_xi(x))
        p = self.w_ip(i)
        return p

    def connect(self, x):
        return self.w_pq(x) # dropout?

    def decode(self, q, t=None, t_one_hot=None):
        j = F.tanh(self.w_qj(q)) 
        y = self.w_jy(j)
        if self.train:
            loss = F.softmax_cross_entropy(y, t)
            q = self.w_yq(t_one_hot)
            return q, loss
        else:
            word = y.data.argmax(1)[0]
            q = self.w_yq(y)
            return q, word 


import unittest
import os
import numpy as np

class TestSeq2Seq(unittest.TestCase):

    def test_init(self):
        src_vocab_size = 100
        src_embed_size = 100
        hidden_size = 100
        dst_mebed_size = 100
        dst_vocab_size = 100
        seq2seq = Seq2Seq(src_vocab_size, src_embed_size, hidden_size, dst_mebed_size, dst_vocab_size)
        seq2seq.reset_state()

    def test_reset_state(self):
        src_vocab_size = 100
        src_embed_size = 100
        hidden_size = 100
        dst_mebed_size = 100
        dst_vocab_size = 100
        seq2seq = Seq2Seq(src_vocab_size, src_embed_size, hidden_size, dst_mebed_size, dst_vocab_size)
        seq2seq.reset_state()
       

if __name__ == "__main__":
    unittest.main()



