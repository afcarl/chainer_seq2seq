#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seq2Seq
"""
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


__authotr__ = "kumada"
__version__ = 0.0
__date__ = ""

class Seq2Seq(chainer.Chain):

    # test ok
    def __init__(self, inout_units, hidden_units):
        """
        @param inout_units: the number of input units 
        @param hidden_units: the number of hidden units 
        """
        super(Seq2Seq, self).__init__(
            l1=L.Linear(inout_units, hidden_units),
            l2=L.LSTM(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, hidden_units),
            l4=L.Linear(hidden_units, inout_units)
        )

        self.train = True

    # test ok
    def reset_state(self):
        self.l2.reset_state()

    def encode(self, x):
        h = self.l1(x)
        p = self.l2(h)
        return p

    def decode(self, p, t=None):
        """
        @param p
        @param t ground truth
        """
        h = self.l3(p) 
        y = self.l4(h)
        p = self.l2(h) 
        if self.train:
            loss = F.mean_squared_error(y, t)
            return p, loss
        else:
            return p, y


import unittest
import os
import numpy as np

class TestSeq2Seq(unittest.TestCase):

    def test_init(self):
        inout_units = 1
        hidden_units = 30
        seq2seq = Seq2Seq(inout_units, hidden_units)
        seq2seq.reset_state()
      

if __name__ == "__main__":
    unittest.main()



