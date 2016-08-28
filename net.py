#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
#from add import add
#import numpy as np
#import math
from chainer import cuda


class Seq2Seq(chainer.Chain):

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
            enc=L.LSTM(src_embed_size, hidden_size), # encoder
            con=L.LSTM(hidden_size, hidden_size), # connector
            w_qj=L.Linear(hidden_size, dst_embed_size),
            w_jy=L.Linear(dst_embed_size, dst_vocab_size),
            dec=L.LSTM(dst_vocab_size, hidden_size) # decoder
        )

        self.train = True

    def reset_state(self):
        self.enc.reset_state()
        self.dec.reset_state()
        self.con.reset_state()

    def encode(self, v):
        i = F.tanh(self.w_xi(v))
        p = self.enc(i)
        return p

    def connect(self, x):
        return self.con(x) # dropout?

    def decode(self, q, t=None, t_one_hot=None):
        j = F.tanh(self.w_qj(q)) 
        y = self.w_jy(j)
        if self.train:
            loss = F.softmax_cross_entropy(y, t)
            q = self.dec(t_one_hot)
            return q, loss
        else:
            word = y.data.argmax(1)[0]
            q = self.dec(y)
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



