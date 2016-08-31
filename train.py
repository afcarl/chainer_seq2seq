#!/usr/bin/env python
# -*- coding: utf-8 -*-

from net import Seq2Seq
import numpy as np
from chainer import optimizers
from chainer import Variable
from chainer import cuda
from utils import *
import cPickle

EPOCHS = 1 
MODEL_PATH = "./model.pkl"
OPTIMIZER_PATH = "./optimizer.pkl"

xp = cuda.cupy

if __name__ == "__main__":

    # load src sequence
    train_src_data = np.array([[1, 2, 3]], dtype=np.float32) 
    print("train_src_data.shape = {s}".format(s=train_src_data.shape))

    # make destination sequence
    train_dst_data = np.fliplr(train_src_data)

    # make a network
    inout_units = 1 
    print("inout_units: {s}".format(s=inout_units))
    hidden_units = 30

    seq2seq = Seq2Seq(
        inout_units, 
        hidden_units 
    )
    seq2seq.reset_state()
    seq2seq.to_gpu()

    # select a optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(seq2seq)

    rows, cols = train_src_data.shape

    # training
    for _ in range(EPOCHS):
        # encode
        for i in range(cols):
            x = Variable(
                xp.asarray(
                    [train_src_data[j, i] for j in range(rows)], 
                    dtype=np.float32
                )[:, np.newaxis],
                volatile="off"
            ) 
            p = seq2seq.encode(x)
        
        # decode
        acc_loss = 0
        for i in range(cols):
            t = Variable(
                xp.asarray(
                    [train_dst_data[j, i] for j in range(rows)], 
                    dtype=np.float32
                )[:, np.newaxis],
                volatile="off"
            )

            p, loss = seq2seq.decode(p, t)
            acc_loss += loss

        seq2seq.zerograds()
        acc_loss.backward()
        acc_loss.unchain_backward()
        optimizer.update()

    # save a model and an optimizer
    cPickle.dump(seq2seq, open(MODEL_PATH, "wb"))
    cPickle.dump(optimizer, open(OPTIMIZER_PATH, "wb"))

