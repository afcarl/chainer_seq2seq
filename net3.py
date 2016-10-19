#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seq2Seq
"""
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer import cuda

xp = cuda.cupy

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
            w_xp=L.Linear(inout_units, 4 * hidden_units),
            w1_pp=L.Linear(hidden_units, 4 * hidden_units),
            w2_pp=L.Linear(hidden_units, 4 * hidden_units),
            w_yp=L.Linear(inout_units, 4 * hidden_units),
            w_py=L.Linear(hidden_units, inout_units)
        )
        self.phase = Seq2Seq.Train
        self.cell_state = None
        self.previous_p = None
        self.hidden_units = hidden_units

    def reset_state(self, batch_size, volatile="off"):
        self.cell_state = Variable(xp.zeros((batch_size, self.hidden_units), dtype=np.float32), volatile=volatile)
        self.previous_p = Variable(xp.zeros((batch_size, self.hidden_units), dtype=np.float32), volatile=volatile)

    def encode(self, x):
        self.cell_state, p = F.lstm(
            self.cell_state, 
            self.w_xp(x) + self.w1_pp(self.previous_p)
        )
        self.previous_p = p
        return p

    def decode(self, p, t=None):
        """
        @param p
        @param t ground truth
        """
        y = self.w_py(p) 
        if self.phase is Seq2Seq.Train:
            loss = F.mean_squared_error(y, t)
            self.cell_state, p = F.lstm(
                self.cell_state,
                self.w_yp(t) + self.w2_pp(self.previous_p)
            )
            self.previous_p = p
            return p, loss
        elif self.phase is Seq2Seq.Valid:
            loss = F.mean_squared_error(y, t)
            self.cell_state, p = F.lstm(
                self.cell_state,
                self.w_yp(y) + self.w2_pp(self.previous_p)
            )
            self.previous_p = p
            return p, loss
        else: # Test
            self.cell_state, p = F.lstm(
                self.cell_state,
                self.w_yp(y) + self.w2_pp(self.previous_p)
            )
            self.previous_p = p
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



