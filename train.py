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
import time

xp = cuda.cupy


def initialize_model(model):
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)


def compute_loss(model, src_data, dst_data, volatile):
    rows, cols = src_data.shape
    indices = np.random.permutation(rows)
    
    # encode
    for i in range(cols):
        x = Variable(
            xp.asarray(
                [src_data[indices[j], i] for j in range(rows)], 
                dtype=np.float32
            )[:, np.newaxis],
            volatile=volatile
        ) 
        p = model.encode(x)
    
    # decode
    acc_loss = 0
    for i in range(cols):
        t = Variable(
            xp.asarray(
                [dst_data[indices[j], i] for j in range(rows)], 
                dtype=np.float32
            )[:, np.newaxis],
            volatile=volatile
        )

        p, loss = model.decode(p, t)
        acc_loss += loss
    return acc_loss


def validate(model, src_data, dst_data):
    validator = model.copy()
    validator.reset_state()
    validator.phase = net.Seq2Seq.Valid
    return compute_loss(validator, src_data, dst_data, "on")

    
def train(train_src_data, valid_src_data):
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

    train_with_pretrained_model(seq2seq, optimizer, train_src_data, valid_src_data)


def train_with_pretrained_model(seq2seq, optimizer, train_src_data, valid_src_data):
    # make destination sequence
    train_dst_data = np.fliplr(train_src_data)
    valid_dst_data = np.fliplr(valid_src_data)

    _, train_cols = train_src_data.shape
    _, valid_cols = valid_src_data.shape
    log_file = open(params.LOG_FILE_PATH, "w")
    start_time = time.time()

    # training
    for epoch in range(1, params.EPOCHS + 1):
        seq2seq.reset_state()
        seq2seq.zerograds()
        acc_loss = compute_loss(seq2seq, train_src_data, train_dst_data, "off")
        acc_loss.backward()
        #acc_loss.unchain_backward()
        optimizer.update()

        if epoch % params.DISPLAY_EPOCH == 0:
            train_loss = acc_loss.data / train_cols
            valid_loss = validate(seq2seq, valid_src_data, valid_dst_data).data / valid_cols
            message = "[{i}]train loss:\t{j}\tvalid loss:\t{k}".format(
                    i=epoch, j=train_loss, k=valid_loss)
            log_file.write(message + "\n")
            log_file.flush()
        

    # save a model and an optimizer
    cPickle.dump(seq2seq, open(params.MODEL_PATH, "wb"))
    cPickle.dump(optimizer, open(params.OPTIMIZER_PATH, "wb"))
    end_time = time.time()
    log_file.write("{s}[m]".format(s=(end_time - start_time)/60))


if __name__ == "__main__":

    # load src sequence
    src_data = np.array([[1, 2, 3], [10, 20, 30], [10, 20, 30]], dtype=np.float32) 
    print("src_data.shape = {s}".format(s=src_data.shape))

    train(src_data)
