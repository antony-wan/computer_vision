"""
Implementation of AlexNet CNN architecture (2012)
AlexNet won the 2012 ImageNet LSVRC-2012
5 convolutionals layers + 3 fully connected layers
Relu applied after each convolutional and fully connected layer
dropout is used before first and second fully connected layer
"""

from Layers import Layers
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, selu, softmax, swish


def AlexNet():

    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    model = Layers()

    x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
    y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
    x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    network=model.conv_layer(layer=x_image, kernel_size=11, input_depth=3, output_depth=96, stride_size=4)

    #55x55x96
    network=model.pooling_layer(layer=network, kernel_size=3, stride_size=2)
    #27x27x96
    network=relu(network)
    print(network)

    #level 2 convolution
    network=model.conv_layer(layer=network, kernel_size=5, input_depth=96, output_depth=256, stride_size=1)
    network=model.pooling_layer(network, kernel_size=3, stride_size=2)
    #13x13x256
    network=relu(network)
    print(network)

    #level 3 convolution
    network=model.conv_layer(layer=network, kernel_size=3, input_depth=256, output_depth=384, stride_size=1)
    network=relu(network)
    print(network)

    #level 4 convolution
    network=model.conv_layer(layer=network, kernel_size=3, input_depth=384, output_depth=384, stride_size=1)
    network=relu(network)
    print(network)

    #level 5 convolution
    network=model.conv_layer(layer=network, kernel_size=3, input_depth=384, output_depth=256, stride_size=1)
    #13x13x256
    network=model.pooling_layer(layer=network, kernel_size=3, stride_size=2)
    #6x6x256
    network=relu(network)
    print(network)

    #flattening layer
    network,features=model.flattening_layer(network)
    print(network)

    tf.nn.dropout(network, keep_prob=True)

    #fully connected layer
    network=model.fully_connected_layer(network,features,4096)
    network=relu(network)
    print(network)

    tf.nn.dropout(network, keep_prob=True)

    #fully connected layer
    network=model.fully_connected_layer(network,4096,4096)
    network=relu(network)
    print(network)

    #output layer
    network=model.fully_connected_layer(network,4096,10)
    #network=tf.nn.softmax(network)
    print(network)

    y_pred = tf.argmax(network, axis=1)

    return x, y, network, y_pred, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate