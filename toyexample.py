# Based on code from https://github.com/tensorflow/cleverhans
#
# This is the code for the paper
#
# Certifying Some Distributional Robustness with Principled Adversarial Training
# Link: https://openreview.net/forum?id=Hk6kPgZA-
#
# Authors: Aman Sinha, Hongseok Namkoong, John Duchi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_mnist import data_mnist
from utils_tf import model_train, model_eval, model_loss
from utils import cnn_model, shallow_model

from keras.models import load_model
from keras.backend import manual_variable_initialization
from attacks import WassersteinRobustMethod
from keras.utils import np_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs',25, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_string('train_dir', '.', 'Training directory')
flags.DEFINE_string('filename_erm', 'erm.h5', 'Training directory')
flags.DEFINE_string('filename_wrm', 'wrm.h5', 'Training directory')

train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
}
eval_params = {'batch_size': FLAGS.batch_size}

seed = 12346
np.random.seed(seed)
tf.set_random_seed(seed)

def toysamples():
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    size = 30000
    Xsample = np.random.multivariate_normal(mean, cov, size)
    Ysample = []
    delidx = []
    for i in range(size):
        l2norm = np.linalg.norm(Xsample[i])

        if l2norm <= np.sqrt(2)/1.3:
            Ysample.append(1)
        elif l2norm >= 1.3*np.sqrt(2):
            Ysample.append(0)
        else:
            delidx.append(i)
    Xsample = np.delete(Xsample, delidx, axis=0)

    # size = int(np.ceil(Xsample.shape[0] / 2))
    # ss = Xsample[size:]
    # posx, posy = ss[np.where(np.array(Ysample[size:]) == 1)].T
    # negx, negy = ss[np.where(np.array(Ysample[size:]) == 0)].T
    #
    # plt.plot(posx, posy, 'o')
    # plt.plot(negx, negy, 'x')
    # plt.axis('equal')
    # plt.show()

    size = int(np.ceil(Xsample.shape[0] / 2))
    Ysample = np_utils.to_categorical(Ysample, 2)
    return Xsample[0:size], Ysample[0:size], Xsample[size:], Ysample[size:]

def main(argv=None):

    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get toy samples
    X_train, Y_train, X_test, Y_test = toysamples()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    # Define TF model graph
    # model = shallow_model(activation='elu')
    # predictions = model(x)
    # wrm = WassersteinRobustMethod(model, sess=sess)
    wrm_params = {'eps': 0.25, 'ord': 2, 'y':y, 'steps': 15}
    # predictions_adv_wrm = model(wrm.generate(x, **wrm_params))
    #
    # def evaluate():
    #     # Evaluate the accuracy of the MNIST model on legitimate test examples
    #     accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
    #     print('Test accuracy on legitimate test examples: %0.4f' % accuracy)
    #
    #     # Accuracy of the model on Wasserstein adversarial examples
    #     accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_wrm, X_test, \
    #                                    Y_test, args=eval_params)
    #     print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)
    #
    # # Train the model
    # model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate, \
    #             args=train_params, save=False)
    # model.model.save(FLAGS.train_dir + '/' + FLAGS.filename_erm)
    # model.save_weights(FLAGS.train_dir + '/' + FLAGS.filename_erm)


    # print('')
    # print("Repeating the process, using Wasserstein adversarial training")
    # # Redefine TF model graph
    # model_adv = shallow_model(activation='elu')
    # predictions_adv = model_adv(x)
    # wrm2 = WassersteinRobustMethod(model_adv, sess=sess)
    # predictions_adv_adv_wrm = model_adv(wrm2.generate(x, **wrm_params))
    #
    # def evaluate_adv():
    #     # Accuracy of adversarially trained model on legitimate test inputs
    #     accuracy = model_eval(sess, x, y, predictions_adv, X_test, Y_test, args=eval_params)
    #     print('Test accuracy on legitimate test examples: %0.4f' % accuracy)
    #
    #     # Accuracy of the adversarially trained model on Wasserstein adversarial examples
    #     # accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_adv_wrm, \
    #     #                                X_test, Y_test, args=eval_params)
    #     # print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)
    #
    # model_train(sess, x, y, predictions_adv_adv_wrm, X_train, Y_train, \
    #             predictions_adv=predictions_adv_adv_wrm, evaluate=evaluate_adv, \
    #             args=train_params, save=False)
    # model_adv.save_weights(FLAGS.train_dir + '/' + FLAGS.filename_wrm)


    print('loading ' + FLAGS.train_dir + '/' + FLAGS.filename_wrm)
    model2 = shallow_model(activation='elu')
    model2.load_weights(FLAGS.train_dir + '/' + FLAGS.filename_wrm)

    # robustness certificate validate
    g = tf.placeholder(tf.float32)
    wrm = WassersteinRobustMethod(model2, sess=sess)
    wrm_params = {'eps': 0.5/g, 'ord': 2, 'y': y, 'steps': 15}
    x_adv = wrm.generate(x, **wrm_params)
    # predictions = model2(x)
    # accuracy = model_eval(sess, x, y, predictions, X_train, Y_train, args=eval_params)
    # print(accuracy)
    robust_surrogate = model_loss(y, model2(x_adv), mean=True)
    rho = 2*tf.nn.l2_loss(x_adv - x)

    with sess.as_default():
        train_rho = []
        train_loss = []
        test_rho = []
        test_loss = []

        for idx in np.arange(1.32, 5 ,0.1):
            gamma = idx

            adv = sess.run(x_adv, feed_dict={x: X_train, y: Y_train, g: gamma})
            # print(adv[0], adv[1])

            posx, posy = adv.T
            plt.plot(posx, posy, 'x')
            plt.axis('equal')
            plt.show()

            certificate, rho_train = sess.run([robust_surrogate, rho],
                                              feed_dict={x: X_train, y: Y_train, g: gamma})
            test_worst_loss, rho_test = sess.run([robust_surrogate, rho],
                                                 feed_dict={x: X_test, y: Y_test, g: gamma})

            # print(X_train.shape[0], X_test.shape[0])
            print(certificate, rho_train/X_train.shape[0], test_worst_loss, rho_test/X_test.shape[0])

            train_rho.append(rho_train/X_train.shape[0])
            train_loss.append(certificate)
            test_rho.append(rho_test/X_test.shape[0])
            test_loss.append(test_worst_loss)

        certificate, rho_train = sess.run([robust_surrogate, rho],
                                          feed_dict={x: X_train, y: Y_train, g: 2.0})
        certificate = certificate - 2*rho_train/X_train.shape[0]
        print(rho_train/X_train.shape[0])

        train_rho = np.arange(0,0.8,0.05)
        plt.plot(train_rho, certificate + train_rho*2, '-r', linewidth=1.0)
        plt.plot(test_rho, test_loss, '-b', linewidth=1.0)
        plt.show()

if __name__ == '__main__':
    app.run()
