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
    Train = 1
    Valid = 2
    Test = 3

    def __init__(self, inout_units, hidden_units):
        """
        @param inout_units: the number of input units 
        @param hidden_units: the number of hidden units 
        """
        super(Seq2Seq, self).__init__(
            l1=L.LSTM(inout_units, hidden_units),
            l2=L.LSTM(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, inout_units)
        )
        self.phase = Seq2Seq.Train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def stacked_lstm(self, x):
        return F.l2(self.l1(x))

    def encode(self, x):
        return self.stacked_lstm(x)

    def decode(self, p, t=None):
        """
        @param p
        @param t ground truth
        """
        y = self.l3(p) 
        if self.phase is Seq2Seq.Train:
            loss = F.mean_squared_error(y, t)
            p = self.stacked_lstm(t) 
            return p, loss
        elif self.phase is Seq2Seq.Valid:
            loss = F.mean_squared_error(y, t)
            p = self.stacked_lstm(y)
            return p, loss
        else: # Test
            p = self.stacked_lstm(y)
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



