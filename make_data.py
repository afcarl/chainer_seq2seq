#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import random


class DataMaker(object):

    def __init__(self):
        pass

    def make(self, one_data, iteration):
        return np.tile(one_data, iteration)

    def load_wave(self, path):
        values =  [float(value.strip()) for value in open(path)]
        interval = 10
        filter = np.ones(interval) / float(interval)
        return np.convolve(values, filter, "valid")

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = i 
            sequences[i] = data[index:index+length_of_sequence]
        return sequences

    def make_mini_batch_without_overlap(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = i * length_of_sequence
            sequences[i] = data[index:index+length_of_sequence]
        return sequences

    def make_mini_batch_without_overlap_for_multi(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence, data.shape[1]), dtype=np.float32)
        for i in range(mini_batch_size):
            index = i * length_of_sequence
            sequences[i] = data[index:index+length_of_sequence,:]
        return sequences

    def make_mini_batch_with_next_steps(self, data, mini_batch_size, length_of_sequence, length_of_prediction):
        sequences = np.ndarray((mini_batch_size, length_of_sequence, length_of_prediction), dtype=np.float32)
        for i in range(mini_batch_size):
            for j in range(length_of_sequence):
                index = i + j + 1
                sequences[i, j] = data[index:index+length_of_prediction]
        return sequences

    def make_mini_batch_with_next_steps_without_overlap(self, data, mini_batch_size, length_of_sequence, length_of_prediction):
        sequences = np.ndarray((mini_batch_size, length_of_sequence, length_of_prediction), dtype=np.float32)
        for i in range(mini_batch_size):
            for j in range(length_of_sequence):
                index = i*length_of_sequence + j + 1
                sequences[i, j] = data[index:index+length_of_prediction]
        return sequences

    def make_mini_batch_with_next_steps_without_overlap_for_multi(self, data, mini_batch_size, length_of_sequence, length_of_prediction):
        sequences = np.ndarray((mini_batch_size, length_of_sequence, length_of_prediction, data.shape[1]), dtype=np.float32)
        for i in range(mini_batch_size):
            for j in range(length_of_sequence):
                index = i*length_of_sequence + j + 1
                sequences[i, j] = data[index:index+length_of_prediction,:]
        return sequences


if __name__ == "__main__":
   pass 
