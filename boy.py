import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 200
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)
"""

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
prediction1 = tf.nn.tanh(tf.matmul(x, W1)+b1)
prediction1_drop = tf.nn.dropout(prediction1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 250], stddev=0.1))
b2 = tf.Variable(tf.zeros([250])+0.1)
prediction2 = tf.nn.tanh(tf.matmul(prediction1_drop, W2)+b2)
prediction2_drop = tf.nn.dropout(prediction2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([250, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(prediction2_drop, W3)+b3)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(201):
        sess.run(tf.assign(lr, 0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        learn_rate = sess.run(lr)
        print("Iter " + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(train_acc) + ",Learning Rate " + str(learn_rate))
