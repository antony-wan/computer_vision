#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:45:59 2019

@author: antony
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image):
    '''
    plot an image from cifar10
    Dimension could be flattened image, ie dimension = (3072,)
    or in rgb, ie (32,32,3)
    '''
    if image.shape == (3072,):
        resized_image = np.reshape(image, [32, 32, 3])
        plt.imshow(resized_image)
    if image.shape == (32,32,3):
        plt.imshow(image)
        
        
def plot_cifar_10(train_x):
    for i in range(4):
        plt.subplot(2,2,i+1)
        plot_image(train_x[i])

