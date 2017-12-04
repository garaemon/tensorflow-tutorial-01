#!/usr/bin/env python
'''
https://www.tensorflow.org/get_started/mnist/pros#build_a_multilayer_convolutional_network
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# Rease mnist dataset. If there is no data in MNIST_data directory,
# automatically download the dataset from web.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '2x2 Convolution layer. padding=SAME means 0 padding.'
    # input of tf.nn.conv2d should be [batch, in_height, in_width, in_channel] tensor.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '2x2 max pooling. '
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 1 layer network is:
#   x = tf.placeholder(tf.float32, [None, 784])
#   W = tf.Variable(tf.zeros([784, 10]))
#   b = tf.Variable(tf.zeros([10]))
#   y = tf.nn.softmax(tf.matmul(x, W) + b)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# input layer. Convert 784 vector to 28x28 image.
# In fact, training data has [None x 784] dimensions.
# convert data to [batchsize, height, width, channel] tensor
# => [-1, 28, 28, 1] for this case.
# tf.reshape function example:
#  1) convert 9 dimensions vector to 3x3 matrix:
#    tf.reshape([1, 2, 3, 4, 5, 6, 7, 8. 9], [3, 3])
#    => [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#  2) flatten tensor.
#    tf.reshape([[[1, 1, 1,], [2, 2, 2]], ...], [-1])
#   => [1, 1, 1, 2, 2, 2, ...]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Build first layer.
# compute 32 features for each 5x5 patch.
# [patch_height, patch_width, input_feature=channel, feature_size] = [5, 5, 1, 32]
W_conv1 = weight_variable([5, 5, 1, 32])
# bias vairables should equal to the number of features(= 32).
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

# Build second layer.
# compute 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Build thrid layer.
# Use fully connected layer for the third layer and outputs 1024 features.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# reshape h_pool2 output tensor to a flat tensor.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

MAX_EPOCHS = 20000
BATCH_SIZE = 50
for i in range(MAX_EPOCHS):
    batch = mnist.train.next_batch(BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0
        })
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(
    'test accuracy: %g',
    accuracy.eval(feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    }))
