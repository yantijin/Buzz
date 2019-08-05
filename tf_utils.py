import numpy as np
import tensorflow as tf


# conv layer
def conv(inputs, output_num, name, kernel_size=5, stride_size=1, init_bias=0.0, conv_padding='SAME', stddev=0.1,
         activation_func=tf.nn.relu, batch_normalization=False):
    with tf.variable_scope(name):
        input_size = inputs.get_shape().as_list()[-1]
        # construct kernal and bias
        conv_weights = tf.get_variable('weights',[kernel_size, kernel_size, input_size, output_num],dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv_biases = tf.get_variable('biases', [output_num], dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=stddev))

        # conv and add bias
        conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, stride_size, stride_size, 1], padding=conv_padding)
        conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
        
        if batch_normalization:
            conv_layer = tf.contrib.layers.batch_norm(conv_layer)

        # activation function
        if activation_func:
            conv_layer = activation_func(conv_layer)
        return conv_layer


def deconv(inputs, output_num, name, kernal_size=5, stride_size=1, deconv_padding='SAME', \
    std_dev=0.1, activation_func=tf.nn.relu, batch_normalization=False):
    with tf.variable_scope(name):
        deconv_layer = tf.layers.conv2d_transpose(inputs, output_num, kernal_size, stride_size, padding=deconv_padding, name=name)
        deconv_bias = tf.get_variable('biases', [output_num], dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=std_dev))
        deconv_layer = tf.nn.bias_add(deconv_layer, deconv_bias)
        
        if batch_normalization:
            deconv_layer = tf.contrib.layers.batch_norm(deconv_layer)

        if activation_func:
            deconv_layer = activation_func(deconv_layer)
        return deconv_layer


# local response normalization
def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)


# 全连接层，如果输入为4维[batch_size, height, width, channels], reshape成 [batch_size, hei*wid*channels]
def fc(inputs, output_size, name, init_bias=0.0, activation_func=tf.nn.relu, stddev=0.1, batch_normalization=False):
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        # flatten the inputs and construct weight and bias
        if len(input_shape) == 4:
            fc_weights = tf.get_variable('weights', [input_shape[1]*input_shape[2]*input_shape[3], output_size], dtype=tf.float32,
                                         initializer=tf.initializers.random_normal(stddev=stddev))
            inputs = tf.reshape(inputs, [input_shape[0], fc_weights.get_shape().as_list()[0]])
        else:
            fc_weights = tf.get_variable('weights', [input_shape[-1], output_size], dtype=tf.float32, 
                                         initializer=tf.initializers.random_normal(stddev=stddev))

        fc_biases = tf.get_variable('biases', [output_size], dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=stddev))
        fc_layer = tf.matmul(inputs, fc_weights)
        fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
        
        if batch_normalization:
            fc_layer = tf.contrib.layers.batch_norm(fc_layer)

        # activate fully connection layer
        if activation_func:
            fc_layer = activation_func(fc_layer)
        return fc_layer

