import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


hidden_layer_1 = 100
minibatch = 50
lmbda = 0.1

curdir = os.path.dirname(os.path.realpath(__file__))


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


mnist = input_data.read_data_sets(os.path.join(curdir, '..', 'data', 'MNIST_data'),
                                  one_hot=True)
  

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(
  tf.truncated_normal([784, hidden_layer_1]))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
W2 = tf.Variable(
  tf.truncated_normal([hidden_layer_1, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Launch the default graph.
sess = tf.Session()
sess.run(tf.initialize_all_variables())

logits = tf.matmul(x, W1) + b1
logits = tf.nn.relu(logits)
logits = tf.matmul(logits, W2) + b2
y = tf.nn.softmax(logits)
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(logits, y_)) + \
      lmbda / minibatch * ( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) )

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
iterations = 15000
for i in range(iterations):
  print(i)
  batch = mnist.train.next_batch(minibatch)
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
