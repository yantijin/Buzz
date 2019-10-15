import tensorflow as tf
import numpy as np
from tf_utils import *
from preprocess import standardize_kpi, complete_timestamp
from math import factorial, pi, pow, log

from tensorflow.python import debug as tf_debug
from tensorflow.contrib.layers import batch_norm, layer_norm

tf.reset_default_graph()


class Buzz():
    def __init__(self, z_dim):
        self._z_dim = z_dim

    @property
    def z_dim(self):
        return self._z_dim

    # `x\in [-10, 10]`
    def build_q_net(self, x, outdimension=64):
        # x->z
        with tf.variable_scope('build_q_net', reuse=tf.AUTO_REUSE):
            lz_x = tf.reshape(x, [-1, 8, 16, 1])

            # 4 conv layers
            lz_x = conv(lz_x, outdimension * 4, 'conv1', stride_size=2, activation_func=tf.nn.leaky_relu,)
            lz_x = conv(lz_x, outdimension * 2, 'conv2', activation_func=tf.nn.leaky_relu)
            lz_x = conv(lz_x, outdimension * 2, 'conv3', stride_size=2, activation_func=tf.nn.leaky_relu)
            lz_x = conv(lz_x, outdimension, 'conv4', activation_func=tf.nn.leaky_relu)
            # get z_mean and z_std by fully connection layer
            # shape [batch_size, z_dimension]
            with tf.name_scope('ZMEAN'):
                z_mean = fc(lz_x, self._z_dim, 'z_mean', activation_func=None)
                # z_mean = tf.layers.batch_normalization(z_mean)
            with tf.name_scope('ZSTD'):
                z_std = fc(lz_x, self._z_dim, 'z_std', activation_func=tf.nn.softplus)
            # shape [batch_size, z_dimension]

            VVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'build_q_net')

            return z_mean, z_std, VVarList

    def genModel(self, z, in_dimension=64, output_num=128):
        with tf.variable_scope('genModel', reuse=tf.AUTO_REUSE):
            batch_size = z.get_shape().as_list()[0]
            # fully connection layer
            lx_z = fc(z, 512, 'fc', activation_func=None)
            # reshape and deconv layers
            lx_z = tf.reshape(lx_z, [-1, 2, 4, in_dimension])

            lx_z = deconv(lx_z, in_dimension * 2, 'deconv1', activation_func=tf.nn.leaky_relu)
            lx_z = deconv(lx_z, in_dimension * 2, 'deconv2', stride_size=2, activation_func=tf.nn.leaky_relu)
            lx_z = deconv(lx_z, in_dimension * 4, 'deconv3', activation_func=tf.nn.leaky_relu)
            lx_z = deconv(lx_z, 1, 'deconv4', stride_size=2, activation_func=tf.nn.leaky_relu)

            resShape = lx_z.get_shape().as_list()
            lx_z = tf.reshape(lx_z, [batch_size, resShape[1] * resShape[2] * resShape[3]])

            lx_z = fc(lx_z, output_num, 'deconvOut', activation_func=None)
            # lx_z = tf.clip_by_value(lx_z, -10, 10)

            GVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'genModel')

            return lx_z, GVarList

    # chain the generative model and q_net
    def autoEncoder(self, x, outDimension=64):
        with tf.name_scope('Variational_Net'):
            mu, sigma, VVarList = self.build_q_net(x, outDimension)

        with tf.name_scope('Z_GenNetInput'):
            z = mu + sigma * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        with tf.name_scope('GenerationNet'):
            Xhat, GVarList = self.genModel(z, outDimension)
        with tf.name_scope('KL_divergence'):
            negKL = 0.5 * tf.reduce_mean(tf.reduce_sum(1 + tf.log(1e-8 + tf.square(sigma)) - \
                                                       tf.square(mu) - tf.square(sigma), 1))

        # shape [batch_size, window_size] , 1
        return Xhat, negKL, VVarList, GVarList

    def discNet(self, x, outdimension=32):
        with tf.variable_scope('discNet', reuse=tf.AUTO_REUSE):
            lz_x = tf.reshape(x, [-1, 8, 16, 1])
            # 4 conv layers
            lz_x = conv(lz_x, outdimension * 4, 'conv1', stride_size=2, activation_func=tf.nn.leaky_relu)
            lz_x = conv(lz_x, outdimension * 2, 'conv2', activation_func=tf.nn.leaky_relu)
            lz_x = conv(lz_x, outdimension * 2, 'conv3', stride_size=2, activation_func=tf.nn.leaky_relu)
            lz_x = conv(lz_x, outdimension, 'conv4', activation_func=tf.nn.leaky_relu)
            dis = fc(lz_x, 1, 'dis', activation_func=None)

            DVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discNet')

            # shape [batch_size, 1]
            return dis, DVarList

    def getLoss(self, x, eta, window_size=128):
        Xhat, negKL, VVarList, GVarList = self.autoEncoder(x)

        #        with tf.variable_scope('DisX'):
        with tf.name_scope('DisX'):
            dis1,DVarList = self.discNet(x)
        #        with tf.variable_scope('DisGenZ'):
        with tf.name_scope('DisGenZ'):
            dis2, DVarList = self.discNet(Xhat)

        with tf.name_scope('wgan_loss'):
            wgan_loss = tf.reduce_mean(dis1 - dis2)  # real - fake

        # get gradient penalty
        batch_size = x.get_shape().as_list()[0]
        kxi = tf.random_uniform(shape=tf.shape(Xhat))
        with tf.name_scope('hatX'):
            Xnew = kxi * (x - Xhat) + Xhat
        with tf.name_scope('get_gradient_penalty'):
            disres,_ = self.discNet(Xnew)
            gradients = tf.gradients(disres, [Xnew])[0]
            # gradients = tf.clip_by_value(gradients, -10, 10)

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradientPenalty = tf.reduce_mean((slopes - 1.) ** 2)  # shape (1)

        # vloss = |wgan_loss| - eta * GP
        with tf.variable_scope('vloss'):
            vloss = tf.math.abs(wgan_loss) - eta * gradientPenalty

        # wloss = -la * |wgan_loss| - KL - logZ
        with tf.variable_scope('omegaLoss', reuse=tf.AUTO_REUSE):
            la = tf.get_variable('lambda', shape=(), initializer=tf.constant_initializer(0.5))
            la = tf.clip_by_value(la, 1e-3, 10)
            logZ = log(factorial(window_size) / factorial(int(window_size / 2)) * 2 * pow(pi,window_size / 2))\
                   - window_size * tf.math.log(la)
            Wloss = -la * tf.math.abs(wgan_loss) + negKL - logZ

        return Xhat, negKL, wgan_loss, gradientPenalty, -vloss, -Wloss, logZ, la, VVarList, GVarList, DVarList


def readData(fileName):
    with open(fileName) as fileObj:
        next(fileObj)
        timestamp = []
        values = []
        labels = []
        for line in fileObj.readlines():
            tmpList = line.split(',')
            timestamp.append(int(tmpList[0]))
            values.append(float(tmpList[1]))
            labels.append(int(tmpList[2]))
        return timestamp, values, labels


if __name__ == "__main__":

    '''
    parameters
    '''
    z_dimension = 13
    eta = 10  # gp weight
    window_size = 128
    n_critic = 3  # number of critic iterations
    s0 = 32  # initial neighborhood size
    b0 = 8  # initial batch size
    # parameters for adam optimizer
    alpha0 = 1e-3
    beta1 = 0.9
    beta2 = 0.999

    # get Data
    fileName = './cpu4.csv'
    timestamp, values, labels = readData(fileName)

    #    labels = np.zeros_like(values, dtype=np.int32)
    # complete the timestamp and obtain the missing point indicators

    timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

    # split data into training and testing data
    test_portion = 0.3
    test_n = int(len(values) * test_portion)

    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    # standardize the training and testing data and clip the data  [-10, 10]
    traibyn_values, mean, std = standardize_kpi(
        train_values, excludes=np.logical_or(train_labels, train_missing))

    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    # build computation graph
    x = tf.placeholder(tf.float32, shape=[s0 * b0, window_size], name='inputData')
    DisOptimizer = tf.train.AdamOptimizer(learning_rate=alpha0, beta1=beta1, beta2=beta2)
    GenOptimizer = tf.train.AdamOptimizer(learning_rate=alpha0, beta1=beta1, beta2=beta2)

    model = Buzz(z_dimension)
    Xhat, negKL, wgan_loss, gradientPenalty, vloss, Wloss, logZ, la, VVarList, GVarList, DVarList = model.getLoss(x, eta)

    VVarList.extend(GVarList)
    VVarList.append(la)



    Dis_loss = DisOptimizer.minimize(vloss, var_list=VVarList)
    Gen_loss = GenOptimizer.minimize(Wloss, var_list=DVarList)

    trainDataLen = len(train_values)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    #    writer = tf.summary.FileWriter('./logs', sess.graph)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)

    epoch = 0
    while s0 > 0:
        # shuffle the data and select omega
        stopNum = (trainDataLen - window_size) // s0  # remove the first window size
        NC = [window_size + i * s0 for i in range(stopNum)]
        np.random.shuffle(NC)
        # print(len(NC))
        # print(stopNum)
        for critic in range(stopNum // b0):

            selectedOmega = NC[b0 * (critic):b0 * (critic + 1)]

            # get input data
            inputs = []
            for i in range(b0):
                for j in range(s0):
                    inputs.append(train_values[selectedOmega[i] + j - window_size + 1: selectedOmega[i] + j + 1])  # [bs, window_size]

            # print(np.shape(inputs))
            if critic % 4 == 3:
                _, GenLoss, WL, GP, _KL, LZ, barX, LAM = sess.run(
                    [Gen_loss, Wloss, wgan_loss, gradientPenalty, negKL, logZ, Xhat, la], feed_dict={x: inputs})
                print('epoch: {} GenLoss: {} s0: {}  b0: {}, wgan_loss: {}, GP: {}, -KL:{}, LZ:{}, LAM:{}'.format(epoch, GenLoss, s0, b0, WL, GP, _KL, LZ, LAM))
            else:
                _, DisLoss, WL, GP, barX, LAM = sess.run([Dis_loss, vloss, wgan_loss, gradientPenalty, Xhat, la],
                                                         feed_dict={x: inputs})
                # print(
                #     'epoch: {} DisLoss: {} s0: {}  b0: {}, wgan_loss: {}, GP: {}, LAM:{}'.format(epoch, DisLoss, s0, b0,
                #                                                                                  WL, GP, LAM))
                # print(barX)
        epoch += 1

        # print('epoch: {}'.format(epoch))
        print('epoch: {} GenLoss: {}  DisLoss: {} s0: {}  b0: {}'.format(epoch, GenLoss, DisLoss, s0, b0))

        if epoch % 40 == 0:
            s0 = s0 // 2
            b0 = b0 * 2
