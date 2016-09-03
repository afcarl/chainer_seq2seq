#!/usr/bin/env python
# -*- coding: utf-8 -*-

import net
import numpy as np
from chainer import optimizers
from chainer import Variable
from chainer import cuda
import cPickle
import params
import sys

xp = cuda.cupy


def initialize_model(model):
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    
def train(src_data):
    # make destination sequence
    dst_data = np.fliplr(src_data)

    # make a network
    seq2seq = net.Seq2Seq(
        params.INOUT_UNITS, 
        params.HIDDEN_UNITS 
    )
    initialize_model(seq2seq)    
    seq2seq.to_gpu()

    # select a optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(seq2seq)

    rows, cols = src_data.shape
    log_file = open(params.LOG_FILE_PATH, "w")

    # training
    for epoch in range(1, params.EPOCHS + 1):
        seq2seq.reset_state()
        seq2seq.zerograds()

        # encode
        for i in range(cols):
            x = Variable(
                xp.asarray(
                    [src_data[j, i] for j in range(rows)], 
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
                    [dst_data[j, i] for j in range(rows)], 
                    dtype=np.float32
                )[:, np.newaxis],
                volatile="off"
            )

            p, loss = seq2seq.decode(p, t)
            acc_loss += loss

        acc_loss.backward()
        #acc_loss.unchain_backward()
        optimizer.update()

        #if epoch != 0 and epoch % params.DISPLAY_EPOCH == 0:
        if epoch % params.DISPLAY_EPOCH == 0:
            train_loss = acc_loss.data / cols
            message = "[{i}]train loss:\t{l}".format(i=epoch, l=train_loss)
            #print(message)
            #sys.stdout.flush()
            log_file.write(message + "\n")
            log_file.flush()
        

    # save a model and an optimizer
    cPickle.dump(seq2seq, open(params.MODEL_PATH, "wb"))
    cPickle.dump(optimizer, open(params.OPTIMIZER_PATH, "wb"))


if __name__ == "__main__":

    # load src sequence
    src_data = np.array([[1, 2, 3], [10, 20, 30], [10, 20, 30]], dtype=np.float32) 
    print("src_data.shape = {s}".format(s=src_data.shape))

    train(src_data)
