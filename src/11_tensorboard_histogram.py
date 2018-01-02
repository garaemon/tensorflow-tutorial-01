#!/usr/bin/env python

import tensorflow as tf

k = tf.placeholder(tf.float32)

mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
tf.summary.histogram('normal/moving_mean', mean_moving_normal)

variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-k)
tf.summary.histogram('normal/shrinking_variance', variance_shrinking_normal)

normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
tf.summary.histogram('normal/bimodal', normal_combined)

gamma = tf.random_gamma(shape=[1000], alpha=k)
tf.summary.histogram('gamma', gamma)

poisson = tf.random_poisson(shape=[1000], lam=k)
tf.summary.histogram('poisson', poisson)

uniform = tf.random_uniform(shape=[1000], maxval=k*10)
tf.summary.histogram('uniform', uniform)

all_distributions = [mean_moving_normal, variance_shrinking_normal, normal_combined, gamma, poisson, uniform]
all_combined = tf.concat(all_distributions, 0)
tf.summary.histogram('all_combined', all_combined)

sess = tf.Session()

if tf.gfile.Exists('/tmp/histogram_example'):
    tf.gfile.DeleteRecursively('/tmp/histogram_example')
tf.gfile.MakeDirs('/tmp/histogram_example')
writer = tf.summary.FileWriter('/tmp/histogram_example')

summaries = tf.summary.merge_all()

N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)
