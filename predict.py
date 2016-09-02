#!/usr/bin/env python
# -*- coding: utf-8 -*-

from net import Seq2Seq
import cPickle
from chainer import Variable
from chainer import cuda
import chainer
import numpy as np
from params import *

xp = cuda.cupy


def predict(model, src_data):
    predictor = model.copy()
    predictor.train = False
    predictor.reset_state()

    rows, cols = src_data.shape
    
    # encode
    for i in range(cols):
        x = Variable(
            xp.asarray(
                [src_data[j, i] for j in range(rows)], 
                dtype=np.float32
            )[:, np.newaxis]
        ) 
        p = predictor.encode(x)

    # decode
    results = np.ndarray((rows, cols), dtype=np.float32)
    for i in range(cols):
        p, y = predictor.decode(p)
        t = chainer.cuda.to_cpu(y.data)
        results[:, i] = chainer.cuda.to_cpu(y.data).reshape((rows,))
    return results


if __name__ == "__main__":

    # load src sequence
    src_data = np.array([[1, 2, 3]], dtype=np.float32) 
    print("src_data.shape = {s}".format(s=src_data.shape))

    # load a trained model
    seq2seq = cPickle.load(open(MODEL_PATH))
    
    results = predict(seq2seq, src_data)
    print(results)

#    rows, cols = src_data.shape
    
#    # encode
#    for i in range(cols):
#        x = Variable(
#            xp.asarray(
#                [src_data[j, i] for j in range(rows)], 
#                dtype=np.float32
#            )[:, np.newaxis]
#        ) 
#        p = seq2seq.encode(x)
#
#    # decode
#    results = np.ndarray((rows, cols), dtype=np.float32)
#    for i in range(cols):
#        p, y = seq2seq.decode(p)
#        t = chainer.cuda.to_cpu(y.data)
#        results[:, i] = chainer.cuda.to_cpu(y.data).reshape((rows,))
