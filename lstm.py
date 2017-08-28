import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


learning_rate = 1e-2
num_epochs = 100
batch_size = 64
display_step = 10

seq_length = 28
num_inputs = 28
num_hidden = 128

num_classes = 10


x = tf.placeholder(tf.float32, [None, seq_length, num_inputs], name='x')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def inference(x, weights, biases):
    x = tf.unstack(x, seq_length, 1)

    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)

    xwb = tf.nn.xw_plus_b(outputs[-1], weights['out'], biases['out'])

    return xwb

pred = inference(x, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


train_loss_summary = tf.summary.scalar('train/loss', loss)
train_acc_summary = tf.summary.scalar('train/accuracy', accuracy)
train_summary_op = tf.summary.merge([train_loss_summary, train_acc_summary])

test_loss_summary = tf.summary.scalar('test/loss', loss)
test_acc_summary = tf.summary.scalar('test/accuracy', accuracy)
test_summary_op = tf.summary.merge([test_loss_summary, test_acc_summary])

writer = tf.summary.FileWriter('log')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    for epoch in range(num_epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, seq_length, num_inputs))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        train_summary = sess.run(train_summary_op, feed_dict={x: batch_x, y: batch_y})
        writer.add_summary(train_summary, epoch)

        if epoch % display_step == 0:
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            print('{}\t[TRAIN]\t\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(datetime.now(), epoch, train_loss, train_acc))

        test_image = mnist.test.images.reshape((-1, seq_length, num_inputs))
        test_label = mnist.test.labels

        test_summary = sess.run(test_summary_op, feed_dict={x: test_image, y: test_label})
        writer.add_summary(test_summary, epoch)

        if epoch % display_step == 0:
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_image, y: test_label})
            print('{}\t [TEST]\t\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(datetime.now(), epoch, test_loss, test_acc))
