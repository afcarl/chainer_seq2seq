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


if __name__ == "__main__":

    # load src sequence
    train_src_vocab = {}
    train_src_data = load_data("./ptb.train.txt", train_src_vocab)
    print("train_src_data.shape = {s}".format(s=train_src_data.shape))
    print("len(train_src_vocab) = {s}".format(s=len(train_src_vocab)))

    # load dst sequence
    train_dst_vocab = {}
    train_dst_data = load_data("./ptb.test.txt", train_dst_vocab)
    print("train_dst_data.shape = {s}".format(s=train_dst_data.shape))
    print("len(train_dst_vocab) = {s}".format(s=len(train_dst_vocab)))

    # make a network
    src_vocab_size = len(train_src_vocab)
    src_embed_size = 100
    hidden_size = 200
    dst_mebed_size = 50
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

    # select a optimizer # inside a loop of epoch?
    optimizer = optimizers.SGD()
    optimizer.setup(seq2seq)

    # training
    for _ in range(EPOCHS):
        # encode
        for word in train_src_data:
            x = Variable(cuda.cupy.array([[word]], dtype=np.int32)) 
            p = seq2seq.encode(x)

        q = seq2seq.connect(p)

        # decode
        acc_loss = 0
        for word in train_dst_data:
            t = Variable(cuda.cupy.array([word], dtype=np.int32))
            
            word_one_hot = make_one_hot(word, dst_vocab_size)
            t_one_hot = Variable(cuda.cupy.array(word_one_hot, dtype=np.float32))

            q, loss = seq2seq.decode(q, t, t_one_hot)
            acc_loss += loss

        seq2seq.zerograds()
        acc_loss.backward()
        acc_loss.unchain_backward()
        optimizer.update()

    # save a model and an optimizer
    cPickle.dump(seq2seq, open(MODEL_PATH, "wb"))
    cPickle.dump(optimizer, open(OPTIMIZER_PATH, "wb"))

