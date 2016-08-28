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


if __name__ == "__main__":

    # load src sequence
    train_src_vocab = {}
    train_src_data = load_data("./ptb.train.txt", train_src_vocab)
    print("train_src_data.shape = {s}".format(s=train_src_data.shape))
    print("len(train_src_vocab) = {s}".format(s=len(train_src_vocab)))

    # load dst sequence
    train_dst_vocab = {}
    train_dst_data = load_data("./ptb.train.txt", train_dst_vocab)
    print("train_dst_data.shape = {s}".format(s=train_dst_data.shape))
    print("len(train_dst_vocab) = {s}".format(s=len(train_dst_vocab)))

    # make a network
    src_vocab_size = len(train_src_vocab)
    src_embed_size = 100
    hidden_size = 100
    dst_mebed_size = 100
    dst_vocab_size = len(train_dst_vocab) 
    seq2seq = Seq2Seq(
        src_vocab_size, 
        src_embed_size, 
        hidden_size, 
        dst_mebed_size, 
        dst_vocab_size
    )
    seq2seq.reset_state()
    seq2seq.to_gpu()

    # select a optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(seq2seq)

    # training
    for _ in range(EPOCHS):
        # encode
        for word in train_src_data:
            x = Variable(cuda.cupy.array([[word]], dtype=np.int32)) 
            p = seq2seq.encode(x)

        q = seq2seq.con(p)

        # decode
        accum_loss = 0
        for word in train_dst_data:
            word_one_hot = make_one_hot(word, dst_vocab_size)
            t = Variable(cuda.cupy.array([word], dtype=np.int32))
            t_one_hot = Variable(cuda.cupy.array(word_one_hot, dtype=np.float32))
            q, loss = seq2seq.decode(q, t, t_one_hot)
            accum_loss += loss

        seq2seq.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()
        optimizer.update()

    # save a model 
    cPickle.dump(seq2seq, open(MODEL_PATH, "wb"))

