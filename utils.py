#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def load_data(filename, vocab):
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset


# test ok
def make_one_hot(w, size):
    base = np.eye(size, dtype=np.float32)
    return base[[w]]


import unittest

class TestUtils(unittest.TestCase):

    def test_load_data(self):
        pass

    def test_make_one_hot(self):
        v = make_one_hot(3, 5)
        a = np.array([[0, 0, 0, 1, 0]])
        self.assertTrue((v == np.array([[0, 0, 0, 1, 0]])).all())

if __name__ == "__main__":
    unittest.main()

