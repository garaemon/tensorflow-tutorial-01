#!/usr/bin/env python
'''
Tutorials code avaialble on https://www.tensorflow.org/get_started/mnist/mechanics.

'''

from argparse import ArgumentParser
import os
import sys
import time
import math

from six.moves import xrange

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# `FLAGS` will be assigned by `ArgumentParser.parse_result()`.
FLAGS = None
IMAGE_SIZE = 28
NUM_CLASSES = 10
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def placeholder_inputs(batch_size):
    '''Generates tensorflow placeholders for input dataset.
    '''
    image_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return (image_placeholder, labels_placeholder)


def fill_feed_dict(data_set, image_placeholder, labels_placeholder):
    'Returns dictonary object for feed_dict argument'
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        image_placeholder: images_feed,
        labels_placeholder: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder,
            data_set):
    'Run evaluation'
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d, Num correct: %d, precision: %0.04f' %
          (num_examples, true_count, precision))


def inference(images, hidden1_units, hidden2_units):
    'Build graph for MNIST model'
    # Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS, hidden1_units],
                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)),
                name='weights'))
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden1_units, hidden2_units],
                stddev=1.0 / math.sqrt(float(hidden2_units)),
                name='weights'))
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    'Loss function'
    # labels is int32. Convert it to int64 because tf.nn.sparse_softmax_cross_entropy_with_logits
    # takes int64 tensor as labels.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    'Training operator'
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    'Evaluate accuracy of `logits`.'
    # Top-1 accuracy
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def run_training():
    'Run training precedureae'
    # read dataset
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # build graph as default graph. All the operators are defined with the default global tf.Graph.
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        # build graph
        logits = inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # loss function
        loss_op = loss(logits, labels_placeholder)

        train_op = training(loss_op, FLAGS.learning_rate)

        eval_correct = evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()
        # Variable initializer operator
        init = tf.global_variables_initializer()

        # object to save checkpoint.
        saver = tf.train.Saver()

        sess = tf.Session()
        # Special object to write summary to a file.
        # `tf.summary.FileWriter.add_summary` should be called.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        # initialize the variables
        sess.run(init)

        # `xrange` is faster than using `range`
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # feed_dict will be the arguments of `tf.Session.run()`.
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder,
                                       labels_placeholder)
            # run a step of training.
            _, loss_value = sess.run([train_op, loss_op], feed_dict=feed_dict)
            # duration in seconds to run a step of training.
            duration = time.time() - start_time

            # print summary and save summary to a file
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                           duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            # Save a checkpoint file.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Result of training data evaluation')
                do_eval(sess, eval_correct, images_placeholder,
                        labels_placeholder, data_sets.train)
                print('Result of validation data evaluation')
                do_eval(sess, eval_correct, images_placeholder,
                        labels_placeholder, data_sets.validation)
                print('Result of test data evaluation')
                do_eval(sess, eval_correct, images_placeholder,
                        labels_placeholder, data_sets.test)


def main(_):
    'Main entrypoint which is called by tf.app.run.'
    if tf.gfile.Exists(FLAGS.log_dir):
        # If FLAGS.log_dir exists already, cleanup the contents in order not to confuse with older
        # experiments.
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Initial larning rate')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer')
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='The number of units in hidden layer 1')
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='The number of units in hidden layer 2')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size. Must devide evenly into the dataset size')
    parser.add_argument(
        '--input-data-dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data'),
        help='Directory to put the input data')
    parser.add_argument(
        '--log-dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR', '/tmp'),
            'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log')
    parser.add_argument(
        '--fake-data',
        default=False,
        action='store_true',
        help='If true, use fake data for unit testing')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[[sys.argv[0]] + unparsed])
