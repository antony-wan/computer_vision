#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:30:41 2019

@author: antony
"""
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from load_cifar import get_data_set


def flip(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x):
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
	
# Produce each image at scaling of 90%, 75% and 60% of original image.
#def zoom(x):
#
#    # Generate 20 crop settings, ranging from a 1% to 20% crop.
#    scales = list(np.arange(0.8, 1.0, 0.01))
#    boxes = np.zeros((len(scales), 4))
#
#    for i, scale in enumerate(scales):
#        x1 = y1 = 0.5 - (0.5 * scale)
#        x2 = y2 = 0.5 + (0.5 * scale)
#        boxes[i] = [x1, y1, x2, y2]
#
#    def random_crop(img):
#        # Create different crops for an image
#        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
#        # Return a random crop
#        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
#
#
#    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
#
#    # Only apply cropping 50% of the time
#    return tf.cond(choice < 1, lambda: x, lambda: random_crop(x))
#


train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")

#def data_augmentation(train_x, train_y, augment_size = 5000):
#    train_size = train_x.shape[0]
#    randidx = np.random.randint(train_size, size=augment_size)
#    x_augmented = train_x[randidx]
#    y_augmented = train_y[randidx]
#    sess = tf.Session()
#
#    augmentations = [flip, color, rotate]
#
#    for i in range(x_augmented.shape[0]):
#        aug_manip = augmentations[np.random.randint(3)]
#        x_i = aug_manip(x_augmented[i].reshape(32,32,3))
#        x_i = sess.run(x_i)
#        x_i = x_i.reshape(-1)
#        x_augmented[i] = x_i
#            
#    train_x = np.concatenate((train_x, x_augmented))
#    train_y = np.concatenate((train_y, y_augmented))
#    return train_x, train_y
#    
#
#train_x, train_y = data_augmentation(train_x, train_y, augment_size = 5000)



def data_augmentation(train_x, train_y, augment_size = 5000):
    train_size = train_x.shape[0]
    randidx = np.random.randint(train_size, size=augment_size)
    x_augmented = train_x[randidx]
    y_augmented = train_y[randidx]
    sess = tf.Session()
    
    x_augmented = tf.convert_to_tensor(x_augmented)
    augmentations = [flip, color, rotate]

    for i in range(x_augmented.shape[0]):
        aug_manip = augmentations[np.random.randint(3)]
        x_i = aug_manip((x_augmented[i].reshape(32,32,3)))
        x_i = sess.run(x_i)
        x_i = x_i.reshape(-1)
        x_augmented[i] = x_i
            
    train_x = np.concatenate((train_x, x_augmented))
    train_y = np.concatenate((train_y, y_augmented))
    return train_x, train_y
    

#train_x, train_y = data_augmentation(train_x, train_y, augment_size = 5000)
#scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60])











