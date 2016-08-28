#!/usr/bin/env python
# -*- coding: utf-8 -*-

from net import Seq2Seq
import cPickle
from utils import *
from chainer import Variable
from chainer import cuda

MODEL_PATH = "./model.pkl"
UPPER_LENGTH = 2 

if __name__ == "__main__":
    # load a trained model
    seq2seq = cPickle.load(open(MODEL_PATH))
    seq2seq.train = False

    # load src sequence
    test_src_vocab = {}
    test_src_data = load_data("./ptb.train.txt", test_src_vocab)
    print("test_src_data.shape = {s}".format(s=test_src_data.shape))
    print("len(test_src_vocab) = {s}".format(s=len(test_src_vocab)))

    # encode
    for word in test_src_data:
        x = Variable(cuda.cupy.array([[word]], dtype=np.int32)) 
        p = seq2seq.encode(x)

    q = seq2seq.con(p)

    # decode
    for _ in range(UPPER_LENGTH):
        q, word = seq2seq.decode(q)


