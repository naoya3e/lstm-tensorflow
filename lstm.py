import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


# MNIST データセット
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# 学習パラメータ
learning_rate = 1e-3
num_epochs    = 30000
batch_size    = 128
display_step  = 10

# ネットワークパラメータ
num_steps   = 28
num_inputs  = 28
num_hidden  = 128
num_classes = 10

# 計算グラフ入力
x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name='x')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')


# Fully Connected
def fully_connected(x, in_size, out_size, name, activation=True):
    with tf.name_scope(name) as scope:
        weight = tf.Variable(tf.random_normal([in_size, out_size]))
        bias   = tf.Variable(tf.random_normal([out_size]))

        xwb = tf.nn.xw_plus_b(x, weight, bias, name=scope)

    if activation:
        return tf.nn.relu(xwb)

    return xwb


# 予測モデル
def inference(x):
    # LSTMに流し込むために shape((batch_size, num_inputs), num_steps) に変形する
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, num_inputs])

    # 1st Layer: fully connected, input shape(?, num_steps, num_inputs)
    fc1 = fully_connected(x, num_inputs, num_hidden, name='fc1')

    # TensorをSequence Tensorに変形しないとLSTMに入力できない
    seq = tf.split(fc1, num_steps, 0)

    # 2nd Layer: LSTM
    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, seq, dtype=tf.float32)

    # 3rd Layer: fully connected
    fc3 = fully_connected(outputs[-1], num_hidden, num_classes, name='fc3', activation=False)

    return fc3


# 予測実行
score = inference(x)

# 損失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# 精度
correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 訓練
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op  = optimizer.minimize(loss)


# 計算グラフ起動
with tf.Session() as sess:
    # 初期化
    sess.run(tf.global_variables_initializer())

    print('\n\n{}    Training start'.format(datetime.now()))

    # 訓練
    for epoch in range(0, num_epochs, batch_size):
        # MNISTのバッチを取得する
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # グラフに流し込むデータバッチを shape(batch_size, 784) -> shape(batch_size, num_steps, num_inputs) に変形する
        batch_x = batch_x.reshape((batch_size, num_steps, num_inputs))

        # ミニバッチ学習
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        if epoch % display_step == 0:
            train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            print('{}      Loss: {:.6f}  Accuracy: {:.6f}'.format(datetime.now(), train_loss, train_accuracy))

    # 評価
    test_data  = mnist.test.images.reshape((-1, num_steps, num_inputs))
    test_label = mnist.test.labels
    test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label})
    print('\n{}    Test start'.format(datetime.now()))
    print('{}      Loss: {:.6f}  Accuracy: {:.6f}'.format(datetime.now(), test_loss, test_accuracy))
