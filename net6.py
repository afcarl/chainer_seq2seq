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
            w1=L.Linear(hidden_units, hidden_units),
            w2=L.Linear(hidden_units, hidden_units),
        )
        self.phase = Seq2Seq.Train
        self.hiddens = []

    def reset_state(self):
        self.l2.reset_state()

    def reset_hiddens(self):
        del self.hiddens[:]

    def encode(self, x):
        h = self.l1(x)
        p = self.l2(h)
        return p 

    @staticmethod
    def calculate_ct(hiddens, p):
        row, col = p.shape # (batch_size, hidden_units)
        s = xp.zeros((row,))
        for hidden in hiddens:
            # hidden: (batch_size, hidden_units)
            s += xp.exp(xp.einsum('...k,...k', p, hidden))

        ct = xp.zeros((row, col))
        for hidden in hiddens:
            a = xp.exp(xp.einsum('...k,...k', p, hidden)) / s
            ct += a[:,xp.newaxis] * hidden 
        return ct

    def decode(self, p, hiddens, t=None):
        """
        @param p
        @param t ground truth
        """
        c = Seq2Seq.calculate_ct(hiddens, p.data)
        vc = Variable(c, dtype=xp.float32)
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
        hidden_units = 30
        seq2seq = Seq2Seq(inout_units, hidden_units)
        seq2seq.reset_state()
        seq2seq.to_gpu()
        v = Variable(xp.zeros((2, inout_units), dtype=xp.float32))
        self.assertEqual((2, inout_units), v.data.shape)
        p = seq2seq.encode(v)
        self.assertEqual((2, hidden_units), p.data.shape)

        a = np.array([[1, 2, 3], [1, 2, 3]])
        b = np.array([[3, 2, 1], [3, 2, 1]])
        c = np.diag(np.inner(a, b))
        d = np.diag(a.dot(b.T))
        e = np.exp(d)
        e += e
        self.assertEqual(c.shape, (a.shape[0],))
        self.assertEqual((c - d).all(), False)
        self.assertEqual((c - np.array([10, 10])).all(), False)

        a = np.array([[1, 2, 3]])
        b = np.array([[3, 2, 1]])
        c = np.diag(np.inner(a, b))
        d = np.diag(a.dot(b.T))
        self.assertEqual(c.shape, (a.shape[0],))
        self.assertEqual((c - d).all(), False)
        self.assertEqual((c - np.array([10])).all(), False)

#        a = np.array([[1, 2, 3], [1, 2, 3]])[:, np.newaxis]
#        b = np.array([[3, 2, 1], [3, 2, 1]])[:, np.newaxis]
#        f = np.inner(a, b)
#        print(f.shape) 
#        print(f)
#        c = np.diag(f)

#        d = np.diag(a.dot(b.T))
#        e = np.exp(d)
#        e += e
#        self.assertEqual(c.shape, (a.shape[0],))
#        self.assertEqual((c - d).all(), False)
#        self.assertEqual((c - np.array([10, 10])).all(), False)
#        print(c)








        hidden = xp.array([[1, 2, 3], [1, 2, 3]])
        hiddens = []
        hiddens.append(hidden)
        hiddens.append(hidden)

        a = np.array([0.5, 0.5])
        b = np.array([[1, 2, 3], [1, 2, 3]])
        c = a[:,np.newaxis] * b 
        d = np.array([[0.5, 1, 1.5], [0.5, 1, 1.5]])
        self.assertEqual((c - d).all(), False)

        p = xp.array([[3, 2, 1], [3, 2, 1]])
        x = Seq2Seq.calculate_ct(hiddens, p)
        y = xp.array([[1, 2, 3], [1, 2, 3]])
        print((x - y).all(), False) 

#        a = Variable(np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)[:, np.newaxis])
#        print(a.data.shape)

if __name__ == "__main__":
    unittest.main()



