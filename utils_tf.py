# Based on code from https://github.com/tensorflow/cleverhans

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import math
import numpy as np
import os
import six
import tensorflow as tf
import time
import warnings

from keras import models
from utils import batch_indices, _ArgsWrapper

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class _FlagsWrapper(_ArgsWrapper):
    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """
    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    #print(op)
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

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
                             padding)), v, u

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
        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
        return sigma, v, u

def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, evaluate=None, lossregfunc=False, regulizer=False, regcons=0.5, model=None, verbose=True, args=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'train_dir'
                 and 'filename'
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        p = 1.0
        loss = ((1-p)*loss + p*model_loss(y, predictions_adv))


    if regulizer:
        if not lossregfunc:
            # collecting from reg loss
            reg_losses = tf.losses.get_regularization_losses()
            loss += tf.add_n(reg_losses)
        else:
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            weights_svd = []
            for w in vars:
                shp = w.get_shape().as_list()
                print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
                if 'kernel' in w.name:
                    # sn = tf.svd(w, compute_uv=False)[..., 0]
                    sn, vv, uu = power_iterate(w, 10, w.name.split('/')[1])
                    weights_svd.append(sn)

            # apply sn by adding to loss
            loss += regcons * tf.add_n(weights_svd)


        # apply sn by adding gradient to backprop
        # grads_and_vars = opt.compute_gradients(loss)
        # new_grads = []
        # for grad, var in grads_and_vars:
        #
        #     shp = var.get_shape().as_list()
        #     print("- {} shape:{} size:{}".format(var.name, shp, np.prod(shp)))
        #     if 'kernel' in var.name:
        #         if len(shp) == 4: # convolutional layer
        #             layer_name = var.name.split('/')[0]
        #             layer = model.get_layer(name=layer_name)
        #             s, left_vector, right_vector = power_iterate_conv(weights=layer.kernel, strides=layer.strides[0],
        #                                          padding=layer.padding.upper(), input_shape=layer.input_shape,
        #                                          output_shape=layer.output_shape, num_iter=10, weight_name=layer_name)
        #             # sn_grad = regcons * s * tf.matmul(left_vector, right_vector, transpose_b=True)
        #         else: # fully connected layer
        #             layer_name = var.name.split('/')[0]
        #
        #             # s, u, v = tf.svd(var)
        #             # left_vector = tf.slice(u, [0, 0], [shp[0],1])
        #             # right_vector = tf.slice(v, [0,0], [shp[1],1])
        #             # sn_grad = regcons * s[0] * tf.matmul(left_vector, right_vector, transpose_b=True)
        #
        #             s, left_vector, right_vector = power_iterate(weights=var, num_iter=10, weight_name=layer_name)
        #             sn_grad = regcons * s * tf.matmul(left_vector, right_vector, transpose_b=True)
        #
        #         grad = tf.add(sn_grad, grad)
        #
        #     new_grads.append( (grad, var) )

    # train_step = opt.apply_gradients(new_grads)
    train_step = opt.minimize(loss)

    with sess.as_default():
        # writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
        if hasattr(tf, "global_variables_initializer"):
            tf.global_variables_initializer().run()
        else:
            sess.run(tf.initialize_all_variables())

        for epoch in six.moves.xrange(args.nb_epochs):
            if verbose:
                print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end]})
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                print("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate()

        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and saved at:" + str(save_path))
        else:
            print("Completed model training.")
        # writer.close()
    return True



def model_train_test(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, evaluate=None, regulizer=False, regcons=0.5, model=None, verbose=True, args=None):

    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    opt1 = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        p = 1.0
        loss = ((1-p)*loss + p*model_loss(y, predictions_adv))


    if regulizer:

        # apply sn by adding gradient to backprop
        # grads_and_vars = opt.compute_gradients(loss)
        # new_grads = []
        # for grad, var in grads_and_vars:
        #
        #     shp = var.get_shape().as_list()
        #     print("- {} shape:{} size:{}".format(var.name, shp, np.prod(shp)))
        #     if 'kernel' in var.name:
        #         if len(shp) == 4:
        #             s, u, v = power_iterate(var, 100)
        #         else:
        #             s, u, v = tf.svd(var)
        #             # s, u, v = power_iterate(var, 10)
        #
        #         left_vector = tf.slice(u, [0, 0], [shp[0],1])
        #         right_vector = tf.slice(v, [0,0], [shp[1],1])
        #         sn_grad = regcons * s[0] * tf.matmul(left_vector, right_vector, transpose_b=True)
        #         grad = tf.add(sn_grad, grad)
        #     new_grads.append( (grad, var) )

        # Collecting loss from reg loss
        reg_losses = tf.losses.get_regularization_losses()
        loss1 = loss + tf.add_n(reg_losses)

        # lossfunc + spectral norm
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_svd = []
        for w in vars:
            shp = w.get_shape().as_list()
            print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
            if 'kernel' in w.name:
                # sn = tf.svd(w, compute_uv=False)[..., 0]
                sn, vv, uu = power_iterate(w, 100, w.name.split('/')[1])
                weights_svd.append(sn)

        # apply sn by adding to loss
        loss2 = loss + regcons*tf.add_n(weights_svd)
        #
        grads_regloss = opt.compute_gradients(loss1)
        grads_lossplusreg = opt1.compute_gradients(loss2)

    # test power iteration
    test = False
    if test:
        if model.name == 'cnn':
            layer1 = model.get_layer(name='conv1')
            weights = layer1.kernel
            strides = layer1.strides[0]
            padding = layer1.padding
            input_shape = layer1.input_shape
            output_shape = layer1.output_shape
            num_iter = 20
            powerit = power_iterate_conv(weights=layer1.kernel, strides=layer1.strides[0], padding=layer1.padding.upper(),
                               input_shape=layer1.input_shape, output_shape=layer1.output_shape, num_iter=10)

            shp = weights.get_shape().as_list()
            w = tf.reshape(weights, [shp[0] * shp[1] * shp[2], shp[3]])
            sn = tf.svd(w, compute_uv=False)[..., 0]
        else: # dense layer
            # layer1 = model.get_layer(name='dense1')
            # weights = layer1.kernel
            weights = tf.placeholder(tf.float32, shape=(300, 784))
            num_iter = 100
            s, u, v = power_iterate(weights, num_iter)
            s1, u1, v1 = tf.svd(weights)



    train_step = opt.minimize(loss)
    # train_step = opt.apply_gradients(new_grads)

    with sess.as_default():
        # writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
        if hasattr(tf, "global_variables_initializer"):
            tf.global_variables_initializer().run()
        else:
            sess.run(tf.initialize_all_variables())

        # Unit test
        # randw = np.random.rand(300,784)
        # layer1 = model.get_layer(name='dense1')
        # randw = layer1.kernel.eval().T
        # for i in range(30):
        #     u, u1, v, v1, s, s1 = sess.run([u, u1, v, v1, s, s1], feed_dict={weights: randw})
        #     randw += np.random.rand(300,784)



        for epoch in six.moves.xrange(args.nb_epochs):
            if verbose:
                print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # grad_soft, grad_soft_svd, grads_regloss = sess.run([grads_and_vars, new_grads, grads_regloss],
                #                                                    feed_dict={x: X_train[start:end],
                #                                                               y: Y_train[start:end]})

                grads_regloss, grads_lossplusreg = sess.run([grads_regloss, grads_lossplusreg],
                                                                   feed_dict={x: X_train[start:end],
                                                                              y: Y_train[start:end]})

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end]})
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                print("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate()

        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and saved at:" + str(save_path))
        else:
            print("Completed model training.")
        # writer.close()
    return True



def model_eval(sess, x, y, model, X_test, Y_test, args=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    # Define symbol for accuracy
    # Keras 2.0 categorical_accuracy no longer calculates the mean internally
    # tf.reduce_mean is called in here and is backward compatible with previous
    # versions of Keras
    acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(y, model))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            #if batch % 100 == 0 and batch > 0:
                #print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            cur_acc = acc_value.eval(
                feed_dict={x: X_test[start:end],
                           y: Y_test[start:end]})

            accuracy += (cur_batch_size * cur_acc)

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy
