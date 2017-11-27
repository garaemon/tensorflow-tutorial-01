#!/usr/bin/env python

# https://www.tensorflow.org/get_started/get_started

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print('tf.constant >')
print(node1, node2)

sess = tf.Session()
print('evaluate tf.constant with Session.run >')
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print('sess.run(adder_node, {a: 3, b: 4.5}): ', sess.run(adder_node, {a: 3, b: 4.5}))
print('sess.run(adder_node, {a: [1, 3], b: [2, 4]}): ',
       sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print('sess.run(add_and_triple, {a: 3, b: 4.5}): ', sess.run(add_and_triple, {a: 3, b: 4.5}))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print('sess.run(linear_model, {x: [1, 2, 3, 4]}): ', sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
# squared_deltas = (linear_model - y)^2
squared_deltas = tf.square(linear_model - y)
# loss = \sum squared_deltas[i]
loss = tf.reduce_sum(squared_deltas)
print('sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ',
    sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print('sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ',
    sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)  # re-initialize
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print('sess.run([W, b, loss]: ', (curr_W, curr_b, curr_loss))

