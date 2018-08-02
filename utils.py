# Based on code from https://github.com/tensorflow/cleverhans

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
import keras.regularizers as regularizers
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import os

if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D


class _ArgsWrapper(object):
    """
    Wrapper that allows attribute access to dictionaries
    """
    def __init__(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)


def save_model(model, dir, filename, weights_only=False):
    """
    Save Keras model
    :param model:
    :param dir:
    :param filename:
    :param weights_only:
    :return:
    """
    # If target directory does not exist, create
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Construct full path
    filepath = os.path.join(dir, filename)

    if weights_only:
        # Dump model weights
        model.save_weights(filepath)
        print("Model weights were saved to: " + filepath)
    else:
        # Dump model architecture and weights
        model.save(filepath)
        print("Model was saved to: " + filepath)


def load_model(directory, filename, weights_only=False, model=None):
    """
    Loads Keras model
    :param directory:
    :param filename:
    :return:
    """

    # If restoring model weights only, make sure model argument was given
    if weights_only:
        assert model is not None

    # Construct full path to dumped model
    filepath = os.path.join(directory, filename)

    # Check if file exists
    assert os.path.exists(filepath)

    # Return Keras model
    if weights_only:
        result = model.load_weights(filepath)
        print(result)
        return model.load_weights(filepath)
    else:
        return keras.models.load_model(filepath)


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def other_classes(nb_classes, class_ind):
    """
    Heper function that returns a list of class indices without one class
    :param nb_classes: number of classes in total
    :param class_ind: the class index to be omitted
    :return: list of class indices without one class
    """

    other_classes_list = list(range(nb_classes))
    other_classes_list.remove(class_ind)

    return other_classes_list


def random_targets(gt, nb_classes):
    """
    Take in the correct labels for each sample and randomly choose target
    labels from the others
    :param gt: the correct labels
    :param nb_classes: The number of classes for this model
    :return: A numpy array holding the randomly-selected target classes
    """
    if len(gt.shape) > 1:
        gt = np.argmax(gt, axis=1)

    result = np.zeros(gt.shape)

    for class_ind in range(nb_classes):
        in_cl = gt == class_ind
        result[in_cl] = np.random.choice(other_classes(nb_classes, class_ind))

    return np_utils.to_categorical(np.asarray(result), nb_classes)


def conv_2d(filters, kernel_shape, strides, padding, input_shape=None, name=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                    space (i.e. the number output of filters in the
                    convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                         the strides of the convolution along the width and
                         height.
    :param padding: (required string) can be either 'valid' (no padding around
                    input or feature map) or 'same' (pad to ensure that the
                    output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
                        layer of the model
    :return: the Keras layer
    """
    if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
        if input_shape is not None:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding,
                          input_shape=input_shape, name=name, )
        else:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding, name=name)
    else:
        if input_shape is not None:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding,
                                 input_shape=input_shape, name=name)
        else:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding, name=name)


def cnn_model_keras(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10, activation='none', name='cnn_keras'):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential(name=name)

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same",
                      input_shape=input_shape, name='conv1'),
              Activation(activation),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid", name='conv2'),
              Activation(activation),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid", name='conv3'),
              Activation(activation),
              Flatten(),
              Dense(nb_classes, name='dense1')]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    #model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model


def cnn_model(x):
    with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
        input_shape = x.get_shape().as_list()
        output_shape = [1, ceil(28/2), ceil(28/2), 64]
        layer_one_reg = spectral_regularizer(scale=0.5, strides=2, padding='same',
                                            input_shape= input_shape, output_shape=output_shape,
                                            weight_name='conv1')
        net = tf.layers.conv2d(inputs=x, kernel_size=(8, 8), strides=2, padding='same',
                         filters=64, activation=tf.nn.elu, name='conv1', kernel_regularizer=layer_one_reg)

        input_shape = output_shape
        output_shape = [1,  ceil((input_shape[1] - (6-1) * 1) / 2), ceil((input_shape[1] - (6-1) * 1) / 2), 128]
        layer_two_reg = spectral_regularizer(scale=0.5, strides=2, padding='valid',
                                             input_shape=input_shape, output_shape=output_shape,
                                             weight_name='conv2')
        net = tf.layers.conv2d(inputs=net, kernel_size=(6, 6), strides=2, padding='valid',
                               filters=128, activation=tf.nn.elu, name='conv2', kernel_regularizer=layer_two_reg)

        input_shape = output_shape
        output_shape = [1, ceil((input_shape[1] - (5 - 1) * 1) / 2), ceil((input_shape[1] - (5 - 1) * 1) / 2), 128]
        layer_three_reg = spectral_regularizer(scale=0.5, strides=2, padding='valid',
                                             input_shape=input_shape, output_shape=output_shape,
                                            weight_name='conv3')
        net = tf.layers.conv2d(inputs=net, kernel_size=(5, 5), strides=1, padding='valid',
                               filters=128, activation=tf.nn.elu, name='conv3', kernel_regularizer=layer_three_reg)
        net = tf.layers.flatten(net)

        layer_four_reg = spectral_regularizer(scale=0.5, weight_name='dense', conv=False)
        model = tf.layers.dense(inputs=net, name='dense',
                      units=10, activation=None, kernel_regularizer=layer_four_reg)

    return model

def dense_model_keras(logits=False, input_ph=None, activation='none', name='dnn_keras'):
    model = Sequential(name=name)

    layers = [Dense(300, input_dim = 784, name='dense1'),
              Activation(activation),
              Dense(100, name='dense2'),
              Activation(activation),
              Dense(10)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model


def dense_model(x):
    with tf.variable_scope('dnn', reuse=tf.AUTO_REUSE):
        nn = tf.layers.dense(x, 300, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))
        nn = tf.layers.dense(nn, 100, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))
        nn = tf.layers.dense(nn, 10, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))

    return nn

def toy_model_keras(logits=False, input_ph=None, model_name=None, activation='none'):
    model = Sequential(name=model_name)

    layers = [Dense(4, input_dim = 2),
              Activation(activation),
              Dense(2),
              Activation(activation),
              Dense(2)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    if logits:
        return model, logits_tensor
    else:
        return model

def spectral_regularizer(
        scale=0.5,
        strides=None,
        padding=None,
        input_shape=None,
        output_shape=None,
        num_iter=10,
        weight_name=None,
        method='power',
        conv=True,
        scope=None):
    """Returns a function that can be used to apply spectral norm regularization to weights.
    Args:
        scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
        scope: An optional scope name.
        method: method to compute spectral norm: svd or power iteration (default)
    Returns:
        A function with signature `spectral_norm(weights)` that apply spectral norm  regularization.
    Raises:
        ValueError: If scale is negative or if scale is not a float.
    """
    if scale < 0.:
        raise ValueError('Setting a scale less than 0 on a regularizer: %g' %scale)
    if scale == 0.:
        logging.info('Scale of 0 disables regularizer.')
        return lambda _: None

    def spectral_norm(weights, name=None):
        """Applies spectral_norm regularization to weights."""
        with ops.name_scope(scope, 'spectral_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor(scale,dtype=weights.dtype.base_dtype,name='scale')

        if method=='power':
            if conv:
                s = power_iterate_conv(weights=weights, strides=strides, padding=padding.upper(),
                                             input_shape=input_shape, output_shape=output_shape, num_iter=num_iter,
                                             weight_name=weight_name)
            else:
                s = power_iterate(weights=weights, num_iter=num_iter, weight_name=weight_name)
            return standard_ops.multiply(my_scale, s, name=name)
        else: # svd
            return standard_ops.multiply(my_scale, standard_ops.svd(weights, compute_uv=False)[..., 0],
                                         name=name)

    return spectral_norm


def power_iterate_conv(
    weights,
    strides,
    padding,
    input_shape,
    output_shape,
    num_iter,
    weight_name,
    u=None,
    ):
    """Perform power iteration for a convolutional layer."""

    strides = [1, strides, strides, 1]
    with tf.name_scope(None, default_name='power_iteration_conv'):
        with tf.variable_scope(weight_name, reuse=tf.AUTO_REUSE):
            u_var = tf.get_variable('u_conv', [1] + list(output_shape[1:]),
                                initializer=tf.random_normal_initializer(),
                                trainable=False)
        u = u_var
        v = None
        for _ in range(num_iter):
            v = tf.nn.conv2d_transpose(u, weights, [1] + list(input_shape[1:]), strides, padding)
            v /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(v), 1e-12))
            u = tf.nn.conv2d(v, weights, strides, padding)
            u /= tf.sqrt(tf.maximum(2 * tf.nn.l2_loss(u), 1e-12))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(u_var,
                             u))
        return tf.reduce_sum(u * tf.nn.conv2d(v, weights, strides,
                             padding))

def power_iterate(weights, num_iter, weight_name):
    """Perform power iteration for a dense layer."""

    with tf.name_scope(None, default_name='power_iteration'):
        w_shape = weights.shape.as_list()
        w = tf.reshape(weights, [-1, w_shape[-1]])

        with tf.variable_scope(weight_name, reuse=tf.AUTO_REUSE):
            u_var = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u = u_var
        v = None
        for i in range(num_iter):
            v = tf.matmul(u, tf.transpose(w))
            v /= (tf.reduce_sum(v ** 2) ** 0.5 + 1e-12)
            u = tf.matmul(v, w)
            u /= (tf.reduce_sum(u ** 2) ** 0.5 + 1e-12)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(u_var,
                              u))
        sigma = tf.reduce_sum(tf.matmul(v, w) * tf.transpose(u))
        return sigma