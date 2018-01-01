#!/usr/bin/env python

'See https://www.tensorflow.org/get_started/summaries_and_tensorboard'

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()



    # y = softmax(Wx + b)
    x = tf.placeholder(tf.float32, [None, 784])

    # classification --> softmax as final layer
    # softmax --> cross entropy as loss function
    # cross_entropy = - \sum y' log(y)
    #   where y is a predicted probability and y' is true answer.
    y_ = tf.placeholder(tf.float32, [None, 10])

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            summay, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,
    help='If true, use fake data for unit testing')
    parser.add_argument('--max_steps', type=int, default=1000,
    help='Number of steps to run trainer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
    help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout')
    parser.add_argument('--data_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
    'tensorflow/mnist/input_data'),
    help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/mnist_with_summaries'),
    help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)