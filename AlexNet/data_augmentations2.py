#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:46:20 2019

@author: antony
"""
from load_cifar import get_data_set
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
#train_x, train_y = get_data_set()
#flipped_img = np.fliplr(train_x)

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image

def random_crop(image, crop_size = np.random.randint(16)+17):
    crop_size = check_size(crop_size)
    h, w, d = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    image = imresize(image,(h,w,d))
    return image

def random_flip(image, rate = 0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image

def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image

def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random'):
    image = np.copy(image_origin)
    if np.random.rand() > p:
        return image
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 32)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


def addNoise(image, amt=0.005):
    noise_mask = np.random.poisson(image*amt)/amt
    noisy_img = image + (noise_mask)
    return np.array(np.clip(noisy_img, a_min=0., a_max=1.), dtype=np.uint8)
    
def data_augmentation(train_x, train_y, augment_size = 1000):
    train_size = train_x.shape[0]
    train_x = train_x.reshape(train_size,32,32,3)
    randidx = np.random.randint(train_size, size=augment_size)
    x_augmented = train_x[randidx]
    y_augmented = train_y[randidx]

    augmentations = [random_flip, random_rotation, random_erasing]

    for i in range(x_augmented.shape[0]):
        aug_manip = augmentations[np.random.randint(3)]
        x_augmented[i] = aug_manip(x_augmented[i])
            
    train_x = np.concatenate((train_x, x_augmented))
    train_y = np.concatenate((train_y, y_augmented))
    return train_x, train_y
    
train_x, train_y = get_data_set()
#train_X, train_Y = data_augmentation(train_x, train_y)

