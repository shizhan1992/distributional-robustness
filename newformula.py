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

from utils_mnist import data_mnist, data_mnist_flat
from utils_tf import model_train, model_eval, model_loss, model_train_adv
from utils import cnn_model, shallow_model, shallow_model_keras

from keras.models import load_model
from keras.backend import manual_variable_initialization
from attacks import WassersteinRobustMethod
from keras.utils import np_utils

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs',15, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
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
    # X_train, Y_train, X_test, Y_test = toysamples()
    # X_train, Y_train, X_test, Y_test = data_mnist_flat()
    X_train, Y_train, X_test, Y_test = data_mnist()

    # Define input TF placeholder
    # x = tf.placeholder(tf.float32, shape=(None, 784))
    # y = tf.placeholder(tf.float32, shape=(None, 10))
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))


    # Define TF model graph ( NOT adv training)

    model = cnn_model(activation='elu')
    # model = shallow_model_keras(activation='elu')
    predictions = model(x)

    # Attackers: WRM---FGSM---IFGM
    wrm = WassersteinRobustMethod(model, sess=sess)
    wrm_params = {'eps': 1.3, 'ord': 2, 'y': y, 'steps': 15}
    predictions_adv_wrm = model(wrm.generate(x, **wrm_params))

    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.1, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    adv_fgsm = fgsm.generate(x, **fgsm_params)
    adv_fgsm = tf.stop_gradient(adv_fgsm)
    preds_adv_fgsm = model(adv_fgsm)

    ifgm = BasicIterativeMethod(model, sess=sess)
    ifgm_params = {'eps': 0.1, 'ord': np.inf, 'eps_iter': 0.02, 'nb_iter': 10, 'clip_min': 0., 'clip_max': 1.}
    adv_ifgm = ifgm.generate(x, **ifgm_params)
    adv_ifgm = tf.stop_gradient(adv_ifgm)
    preds_adv_ifgm = model(adv_ifgm)

    pgm = MadryEtAl(model, sess=sess)
    pgm_params = {'eps': 0.1, 'ord': np.inf, 'eps_iter': 0.01, 'nb_iter': 30, 'clip_min': 0., 'clip_max': 1.}
    adv_pgm = pgm.generate(x, **pgm_params)
    adv_pgm = tf.stop_gradient(adv_pgm)
    preds_adv_pgm = model(adv_pgm)

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

        # Accuracy of the model on Wasserstein adversarial examples
        # accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_wrm, X_test, \
        #                                Y_test, args=eval_params)
        # print('Test accuracy on Wasserstein examples: %0.4f' % accuracy_adv_wass)

        # Accuracy of the model on FGSM adversarial examples
        accuracy_adv_fgsm = model_eval(sess, x, y, preds_adv_fgsm, X_test, \
                                       Y_test, args=eval_params)
        print('Test accuracy on fgsm examples: %0.4f' % accuracy_adv_fgsm)

        # Accuracy of the model on IFGM adversarial examples
        accuracy_adv_ifgm = model_eval(sess, x, y, preds_adv_ifgm, X_test, \
                                       Y_test, args=eval_params)
        print('Test accuracy on ifgm examples: %0.4f' % accuracy_adv_ifgm)

        # Accuracy of the model on IFGM adversarial examples
        accuracy_adv_pgm = model_eval(sess, x, y, preds_adv_pgm, X_test, \
                                       Y_test, args=eval_params)
        print('Test accuracy on pgm examples: %0.4f\n' % accuracy_adv_pgm)

    # Train the model
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate, \
                args=train_params, save=False)
    model_train_adv(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate, \
                        args=train_params, save=False)


    # print('')
    # print("Repeating the process, using Wasserstein adversarial training")
    # # Redefine TF model graph (adv training)
    # predictions_adv = shallow_model(x)
    #
    # # Attackers
    # wrm2 = WassersteinRobustMethod(shallow_model, sess=sess)
    # wrm_params = {'eps': 1.3, 'ord': 2, 'y': y, 'steps': 15}
    # predictions_adv_adv_wrm = shallow_model(wrm2.generate(x, **wrm_params))
    #
    # fgsm2 = FastGradientMethod(shallow_model, sess=sess)
    # fgsm_params = {'eps': 0.2, 'clip_min': 0., 'clip_max': 1.}
    # preds_adv_adv_fgsm = shallow_model(tf.stop_gradient(fgsm2.generate(x, **fgsm_params)))
    #
    # ifgm2 = BasicIterativeMethod(shallow_model, sess=sess)
    # ifgm_params = {'eps': 0.2, 'eps_iter': 0.05, 'nb_iter': 10, 'clip_min': 0., 'clip_max': 1.}
    # preds_adv_adv_ifgm = shallow_model(tf.stop_gradient(ifgm2.generate(x, **ifgm_params)))
    #
    # def evaluate_adv():
    #     # Accuracy of adversarially trained model on legitimate test inputs
    #     accuracy = model_eval(sess, x, y, predictions_adv, X_test, Y_test, args=eval_params)
    #     print('Test accuracy on legitimate test examples: %0.4f' % accuracy)
    #
    #     # Accuracy of the adversarially trained model on Wasserstein adversarial examples
    #     # accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_adv_wrm, \
    #     #                                X_test, Y_test, args=eval_params)
    #     # print('Test accuracy on Wasserstein examples: %0.4f' % accuracy_adv_wass)
    #
    #     # Accuracy of the model on Wasserstein adversarial examples
    #     accuracy_adv_fgsm = model_eval(sess, x, y, preds_adv_adv_fgsm, X_test, \
    #                                    Y_test, args=eval_params)
    #     print('Test accuracy on fgsm examples: %0.4f' % accuracy_adv_fgsm)
    #
    #     # Accuracy of the model on IFGM adversarial examples
    #     accuracy_adv_ifgm = model_eval(sess, x, y, preds_adv_adv_ifgm, X_test, \
    #                                    Y_test, args=eval_params)
    #     print('Test accuracy on ifgm examples: %0.4f\n' % accuracy_adv_ifgm)
    #
    #
    # model_train_adv(sess, x, y, predictions_adv, X_train, Y_train, evaluate=evaluate_adv, \
    #             args=train_params, save=False)


if __name__ == '__main__':
    app.run()
