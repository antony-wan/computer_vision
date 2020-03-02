#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:11:58 2019

@author: antony
"""

import tensorflow as tf


class Layers:

    def add_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

    def add_biases(self, shape):
        return tf.Variable(tf.constant(0.05, shape=shape))

    #stride : distance between the receptive field centers of neighboring neurons in a kernel map
    def conv_layer(self, layer, kernel_size, input_depth, output_depth, stride_size):
        weights = self.add_weights([kernel_size, kernel_size, input_depth, output_depth])
        biases = self.add_biases([output_depth])
        stride = [1, stride_size, stride_size, 1]
        layer = tf.nn.conv2d(layer, weights, strides=stride, padding='SAME') + biases
        return layer

    def pooling_layer(self,layer, kernel_size, stride_size):
        kernel = [1, kernel_size, kernel_size, 1]
        stride = [1, stride_size, stride_size, 1]
        return tf.nn.max_pool(layer, ksize=kernel, strides=stride, padding='SAME')

    def flattening_layer(self, layer):
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]), new_size

    def fully_connected_layer(self, layer, input_depth, output_depth):
        weights = self.add_weights([input_depth, output_depth])
        biases = self.add_biases([output_depth])
        layer = tf.matmul(layer, weights) + biases  # mX+b
        return layer

    def activation_layer(self, layer):
        return tf.nn.relu(layer)

#    def batch_norm(self, layer):
#        mean, var = tf.nn.moments(layer, [0, 1, 2])
#        gamma = tf.Variable()
#        beta = tf.Variable()
#        return tf.nn.batch_normalization(layer, mean, var, beta, gamma, epsilon)
#    
    
    
    
    
    
    
    
    