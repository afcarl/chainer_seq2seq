#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seq2Seq
"""
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I


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
        initializer = I.HeNormal()
        super(Seq2Seq, self).__init__(
            l1=L.Linear(inout_units, hidden_units, initialW=initializer),
            l2=L.LSTM(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, inout_units, initialW=initializer),
            w1=L.Linear(hidden_units, hidden_units, initialW=initializer),
            w2=L.Linear(hidden_units, hidden_units, initialW=initializer),
        )
        self.phase = Seq2Seq.Train
        self.train = True

    def reset_state(self):
        self.l2.reset_state()

    def set_phase(self, phase):
        self.phase = phase
        if phase == Seq2Seq.Train:
            self.train = True
        else:
            self.train = False 

    def encode(self, x):
        h = self.l1(x)
        p = self.l2(h)
        return p 

    # test ok
    @staticmethod
    def calculate_ct(hiddens, p):
        batch_size, hidden_units = p.shape
        s = xp.zeros((batch_size,))
        fun = lambda x, y: xp.exp(xp.sum(x * y, axis=1))
        for hidden in hiddens:
            s += fun(p, hidden)

        ct = xp.zeros((batch_size, hidden_units))
        for hidden in hiddens:
            # a.shape == (batch_size,)
            # hidden == (batch_size, hidden_units)
            a = fun(p, hidden) / s
            ct += a[:,xp.newaxis] * hidden 
        return ct.astype(xp.float32)

    def decode(self, p, hiddens, t=None):
        """
        @param p
        @param t ground truth
        """
        c = Seq2Seq.calculate_ct(hiddens, p.data)
        vc = Variable(c)
        q = F.tanh(self.w1(vc) + self.w2(p))
        y = self.l3(q) 
        if self.phase is Seq2Seq.Train:
            loss = F.mean_squared_error(y, t)
            p = self.encode(y)
            return p, loss
        elif self.phase is Seq2Seq.Valid:
            loss = F.mean_squared_error(y, t)
            p = self.encode(y)
            return p, loss
        else: # Test
            p = self.encode(y)
            return p, y 



import unittest
import os
import numpy as np
from chainer import cuda

xp = cuda.cupy

class TestSeq2Seq(unittest.TestCase):

    def test_(self):
        inout_units = 1
        hidden_units = 3
        seq2seq = Seq2Seq(inout_units, hidden_units)
        seq2seq.reset_state()
        seq2seq.to_gpu()
        v = Variable(xp.zeros((2, inout_units), dtype=xp.float32))
        self.assertEqual((2, inout_units), v.data.shape)
        p = seq2seq.encode(v)
        self.assertEqual((2, hidden_units), p.data.shape)

        batch_size = 2
        a = xp.arange(6).reshape(batch_size, hidden_units)
        b = xp.sum(a * a, axis=1)
        self.assertEqual((batch_size,), b.shape)
        self.assertEqual((xp.array([5,50]) - b).all(), False)     

        a = xp.arange(0, 6).reshape(batch_size, hidden_units)
        b = xp.arange(1, 7).reshape(batch_size, hidden_units)
        c = xp.sum(a * b, axis=1)
        self.assertEqual((xp.array([8,62]) - c).all(), False)     

        hidden0 = xp.arange(0, 6).reshape(batch_size, hidden_units)
        hidden1 = xp.arange(1, 7).reshape(batch_size, hidden_units)
        hiddens = [hidden0, hidden1]

        p = xp.arange(2, 8).reshape(batch_size, hidden_units)
        x = Seq2Seq.calculate_ct(hiddens, p)

        a = [0.99987661, 1.99987661, 2.99987661, 3.99999998, 4.99999998, 5.99999998]
        a = xp.array(a).reshape(batch_size, hidden_units)
        b = xp.abs(a - x)
        self.assertEqual((b < 1.0e-06).all(), True)
        self.assertEqual((batch_size, hidden_units), x.shape)

if __name__ == "__main__":
    unittest.main()



