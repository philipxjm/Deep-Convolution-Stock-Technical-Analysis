import tensorflow as tf
import math
import numpy as np
import functools as ft

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

def conv1d(input, output_dim,
           conv_w=9, conv_s=2,
           padding="SAME", name="conv1d",
           stddev=0.02, bias=False):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [conv_w, input.get_shape().as_list()[-1], output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    c = tf.nn.conv1d(input, w, conv_s, padding=padding)

    if bias:
      b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
      return c + b

    return c

def conv2d(input, output_dim,
           conv_h=5, conv_w=5, conv_s=1,
           padding="SAME", name="conv2d",
           stddev=0.02, bias=True):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [conv_h, conv_w, input.get_shape().as_list()[-1], output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    c = tf.nn.conv2d(input, w, strides=[1, conv_s, conv_s, 1], padding=padding)

    if bias:
      b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
      return c + b

    return c

def deconv2d(input, output_dim,
             conv_h=5, conv_w=5, conv_s=1,
             padding="SAME", name="deconv2d",
             stddev=0.02, bias=False):
  with tf.variable_scope(name):
    batch, input_h, input_w, input_dim = [int(d) for d in input.get_shape()]
    w = tf.get_variable('w', [conv_h, conv_w, output_dim, input_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    c = tf.nn.conv2d_transpose(input, w, output_shape=[batch, input_h*2, input_w*2, output_dim],
      strides=[1, conv_s, conv_s, 1], padding=padding)

    if bias:
      b = tf.get_variable('b', [input_dim], initializer=tf.constant_initializer(0.0))
      return c + b

    return c

def batchnorm(input, name="batchnorm", is_2d=False):
  with tf.variable_scope(name):
    input = tf.identity(input)
    channels = input.get_shape()[-1]
    offset = tf.get_variable("offset", [channels],
                              dtype=tf.float32,
                              initializer=tf.zeros_initializer())
    scale = tf.get_variable("scale", [channels],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(1.0, 0.02))
    if is_2d:
      mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
    else:
      mean, variance = tf.nn.moments(input, axes=[0, 1], keep_dims=False)
    variance_epsilon = 1e-5
    normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                           variance_epsilon=variance_epsilon)
    return normalized

def max_pool(input, ksize=2, pool_stride=2, padding="SAME", name="max_pool"):
  with tf.variable_scope(name):
    return tf.nn.max_pool(
        input,
        ksize=[1,ksize,ksize,1],
        strides=[1,pool_stride,pool_stride,1],
        padding=padding)

def fully_connected(input, output_dim, name="fc", stddev=0.02):
  with tf.variable_scope(name):
    unfolded_dim = ft.reduce(lambda x, y: x*y, input.get_shape().as_list()[1:])
    w = tf.get_variable('w',
      [unfolded_dim, output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
    input_flat = tf.reshape(input, [-1, unfolded_dim])

    return tf.matmul(input_flat, w) + b

def weight_variable(shape, name="weight"):
  with tf.variable_scope(name):
    randomized_initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.get_variable('w', [k_h, k_w, input_.get_shape().as_list()[-1], output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))

def bias_variable(shape, name="bias"):
  with tf.variable_scope(name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lrelu(x, a):
  with tf.name_scope("lrelu"):
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def relu(x, name="relu"):
  return tf.nn.relu(x)
