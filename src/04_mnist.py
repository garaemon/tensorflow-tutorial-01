#!/usr/bin/env python
'''
MNIST is a dataset of the images of handwritten digits.
In this tutorials, use softmax to classify the images.abs

MNIST dataset:

MNIST dataset is consist of three parts:
  1. training data (55,000 data points)
  2. test data (10,000 data points)
  3. validation data (5,000 data points)

Each image of MNIST dataset is a 28x28 image and it is represented as
28x28 = 784 dimensions array.


y = softmax(Wx + b)
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# Rease mnist dataset. If there is no data in MNIST_data directory,
# automatically download the dataset from web.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# y = softmax(Wx + b)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# classification --> softmax as final layer
# softmax --> cross entropy as loss function
# cross_entropy = - \sum y' log(y)
#   where y is a predicted probability and y' is true answer.
y_ = tf.placeholder(tf.float32, [None, 10])
# use reduce_sum to sum the second dimension elements.
# use reduce_mean to computes the mean over all the examples of the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# tf.argmax gives the index of the highest entry in tensor along some axis.
# tf.argmax(y, 1) means the index of the highest entry in the second elements of y.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
print('Run 1000 epochs with batch size = 100')
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print('accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
