#!/usr/bin/env python
# -*- coding: utf-8 -*-

from net import Seq2Seq
import cPickle
from chainer import Variable
from chainer import cuda
import chainer
import numpy as np

MODEL_PATH = "./model.pkl"

xp = cuda.cupy

if __name__ == "__main__":

    # load src sequence
    src_data = np.array([[1, 2, 3], [10, 20, 30], [10, 20, 30]], dtype=np.float32) 
    print("src_data.shape = {s}".format(s=src_data.shape))

    # load a trained model
    seq2seq = cPickle.load(open(MODEL_PATH))
    seq2seq.train = False

    rows, cols = src_data.shape
    
    # encode
    for i in range(cols):
        x = Variable(
            xp.asarray(
                [src_data[j, i] for j in range(rows)], 
                dtype=np.float32
            )[:, np.newaxis]
        ) 
        p = seq2seq.encode(x)

    # decode
    results = np.ndarray((rows, cols), dtype=np.float32)
    for i in range(cols):
        p, y = seq2seq.decode(p)
        t = chainer.cuda.to_cpu(y.data)
        results[:, i] = chainer.cuda.to_cpu(y.data).reshape((rows,))
