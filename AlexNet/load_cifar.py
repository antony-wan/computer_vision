"""
Load images from data_set/batches.meta and returs data and label

"""

import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import tensorflow as tf


def get_data_set(name="train"):
    '''
    Parameters :
    ------
    name : string
        name = "train" or name = "test"    
        
    Return :
    ------
    data, label : array
        size(data) = [numbers of images, dimension of an image]
        size(label) = [numbers of images, one hot corresponding to label]
    
    Example :
        
    '''
    x = None
    y = None

    f = open('./data_set/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0

            _X = _X.reshape([-1, 3, 32, 32])
            #(10000, 3,32,32)
            _X = _X.transpose([0, 2, 3, 1])   
            _X = _X.reshape(-1, 32*32*3)
            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot