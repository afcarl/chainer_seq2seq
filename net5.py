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

    # test ok
    def __init__(self, inout_units, hidden_units):
        """
        @param inout_units: the number of input units 
        @param hidden_units: the number of hidden units 
        """
        super(Seq2Seq, self).__init__(
            l1=L.Linear(inout_units, hidden_units)
            l2=L.LSTM(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, inout_units)
        )
        self.phase = Seq2Seq.Train

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
        y = self.l3(p) 
        if self.phase is Seq2Seq.Train:
            loss = F.mean_squared_error(y, t)
            h = self.l1(t) 
            p = self.l2(h)
            return p, loss
        elif self.phase is Seq2Seq.Valid:
            loss = F.mean_squared_error(y, t)
            h = self.l1(y)
            p = self.l2(h)
            return p, loss
        else: # Test
            h = self.l1(y)
            p = self.l2(h)
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



