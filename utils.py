#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
utilities
"""
__author__ = "kumada"
__version__ = "0.0"
__date__ = ""


from chainer import cuda
import numpy as np
import time
import chainer
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.spatial
from make_data import *


xp = cuda.cupy  


def calculate_differences(real_data, prediction_data, next_index):
    """
    @param real_data np.array with the shape of (seq, dim)
    @param prediction_data np.array with the shape of (pred, seq, dim)
    @param next_index `next_index = 1` indicates 1-step prediction.
    @return differences between observations and predictions
    """

    diffs = np.ndarray((len(prediction_data[next_index]), real_data.shape[1]))
    start = time.time()
    # prediction_data[next_index]:(seq, dim)
    for i, (pred, real) in enumerate(zip(prediction_data[next_index], real_data[next_index:])):
        diff = pred - real
        diffs[i, :] = diff
    end = time.time()
    print("{t}[ms/one point]".format(t=1000*(end - start)/prediction_data.shape[1]))
    return diffs


def predict_next_steps(model, data, prediction_length, dim):
    """
    @param model a pre-trainded model
    @param data 
    @return predictions with type of (pred, seq)
    """

    predictor = model.copy()
    predictor.train = False
    data_length = len(data)

    # (pred, seq, dim)
    predictions = np.ndarray((prediction_length + 1,  len(data) - prediction_length, dim))
    predictor.reset_state()
    
    for i, value in enumerate(data[:data_length - prediction_length]):
        x = chainer.Variable(xp.asarray([value], dtype=np.float32)[:, np.newaxis]) # (dim, np.newaxis)
        t = predictor(x,  x) # (1, dim*pred)
        predictions[0, i] = value
        
        for k in range(t.data.shape[1]/dim): 
            predictions[k+1,  i] = t.data[0:, dim*k:dim*k+dim].reshape((dim,)).tolist()
    return predictions


def scale_data(dataset):
    """
    @param dataset
    @return a scaled data
    """
    # add a new axis
    dataset = dataset[:, np.newaxis]

    # normalize dataset to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset


def train_loss_generator(path):
    """
    This function generates training losses.
    @param path a path to a log file
    """
    for line in open(path):
        tokens = line.strip().split()
        if len(tokens) == 8:
            yield math.log10(float(tokens[2]))

            
def test_loss_generator(path):
    """
    This function generates testing losses.
    @param path a path to a log file
    """
    for line in open(path):
        tokens = line.strip().split()
        if len(tokens) == 7:
            yield math.log10(float(tokens[5]))


def make_diff_matrix(pred, diffs_1, diffs_2, diffs_3, prediction_length, dim):
    """
    This function makes a matrix which consists of three difference vectors
    @param pred predictions which are calculated by means of the pre-trained model
    @param diffs_1 differences for 1-step predictions
    @param diffs_2 differences for 2-step predictions
    @param diffs_3 differences for 3-step predictions
    @param prediction_length
    @return matrix
    """
    # pred: (pred, seq, dim)
    # diff_matrix: (seq, pred, dim)
    diff_matrix = np.ndarray((pred.shape[1] - prediction_length + 1, prediction_length, dim))
    
    (rows, cols, _) = diff_matrix.shape
    for i in range(rows):
        v0 = diffs_1[i+2] # (dim,)
        v1 = diffs_2[i+1]
        v2 = diffs_3[i+0]
        diff_matrix[i] = np.array([v0, v1, v2]) # (pred, dim)
    return diff_matrix


def calculate_mean_and_cov(matrix):
    """
    This function calculates a mean and a covariance.
    @param matrix
    @return (mean, cov)
    """
    mean = np.mean(matrix, axis=0)
    cov = np.cov(matrix, rowvar=0)
    return (mean, cov)


def convert_to_mahalanobis_distances(mean, cov, diffs):
    """
    This function calculates mahalanobis distances.
    @param mean a mean
    @param cov a covariance
    @param diffs a difference matrix 
    @return an array of distances
    """

    inv_cov = np.linalg.inv(cov)
    return np.array([scipy.spatial.distance.mahalanobis(diff, mean, inv_cov) for diff in diffs])


def norm_pdf_multivariate(x, mu, sigma):
    """
    This function calculates a multivariate gaussian function.
    @param x 
    @param mu a mean 
    @param sigma a covariance
    @return result
    """

    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

#        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
#        return norm_const * result
        return result
    else:
        raise NameError("The dimensions of the input don't match")


def log_norm_pdf_multivariate(x, mu, sigma):
    """
    This function calculates log10 of a likelihood.
    """
    return np.log10(norm_pdf_multivariate(x, mu, sigma))


def calculate_likelihood(diffs, mu, sigma):
    return np.array([ log_norm_pdf_multivariate(diff, mu, sigma)  for diff in diffs])    


def make_mini_batch_without_overlap_for_multi(dataset, length_of_sequence):
    """
    This function makes a set of mini baches.
    @param dataset array with (seq, dim)
    @param length_of_sequence
    @return a tensor with (mini_batch_size, seq, dim)
    """

    dataset_length = dataset.shape[0]
    mini_batch_size =  dataset_length/length_of_sequence
    data_maker = DataMaker()
    return  data_maker.make_mini_batch_without_overlap_for_multi(
        dataset, 
        mini_batch_size=mini_batch_size, 
        length_of_sequence=length_of_sequence)


def make_mini_batch_without_overlap(dataset, length_of_sequence):
    """
    This function makes a set of mini baches.
    @param dataset array with (seq, dim)
    @param length_of_sequence
    @return a tensor with (mini_batch_size, seq, dim)
    """

    dataset_length = dataset.shape[0]
    mini_batch_size =  dataset_length/length_of_sequence
    data_maker = DataMaker()
    return  data_maker.make_mini_batch_without_overlap(
        dataset, 
        mini_batch_size=mini_batch_size, 
        length_of_sequence=length_of_sequence)


def make_mini_batch_with_next_steps_without_overlap_for_multi(dataset, length_of_sequence, prediction_length):
    """
    This function makes a set of mini baches.
    @param dataset array with (seq, dim)
    @param length_of_sequence
    @param prediction_length 
    @return a tensor with (mini_batch_size, seq, pred, dim)
    """

    dataset_length = dataset.shape[0]
    mini_batch_size =  dataset_length/length_of_sequence
    data_maker = DataMaker()
    return data_maker.make_mini_batch_with_next_steps_without_overlap_for_multi(
        dataset, 
        mini_batch_size=mini_batch_size, 
        length_of_sequence=length_of_sequence,
        length_of_prediction=prediction_length)

 
if __name__ == "__main__":
    pass

