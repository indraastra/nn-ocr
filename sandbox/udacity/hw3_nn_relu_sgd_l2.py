import random
import numpy as np
import tensorflow as tf
from six.moves import range

from matlab_port.display_data import display_data
from matlab_port.utils import load_data, partition_data, shuffle_data

from .dataset import *


batch_size = 50
hidden_layer_1 = 1024
num_steps = 3001
beta = 0.01


def gen_feedforwarder(weights_1, biases_1, weights_2, biases_2):
  def feedforwarder(data, dropout=False):
    logits_1 = tf.matmul(data, weights_1) + biases_1
    output_1 = tf.nn.relu(logits_1)
    if dropout: output_1 = tf.nn.dropout(output_1, .5)
    logits_2 = tf.matmul(output_1, weights_2) + biases_2
    return logits_2
  return feedforwarder


graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer_1]))
  biases_1 = tf.Variable(tf.zeros([hidden_layer_1]))
  weights_2 = tf.Variable(
    tf.truncated_normal([hidden_layer_1, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  feedforwarder = gen_feedforwarder(weights_1, biases_1, weights_2, biases_2)
  logits = feedforwarder(tf_train_dataset, dropout=True)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
        beta * ( tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) )
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(feedforwarder(tf_valid_dataset))
  test_prediction = tf.nn.softmax(feedforwarder(tf_test_dataset))


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = 0
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))

  pred_test_labels = test_prediction.eval()
  print("Test accuracy: %.1f%%" % accuracy(pred_test_labels, test_labels))

  predictions = np.argmax(pred_test_labels, 1)
  labels = np.argmax(test_labels, 1) 
  error_idx = predictions != labels

  # Randomly select 100 data points to display
  error_test_data = test_dataset[error_idx, :]
  predictions = predictions[error_idx]
  labels = labels[error_idx]
  m = error_test_data.shape[0]
  sel = random.sample(range(m), 100)

  print("Predicted: ")
  print(predictions[sel].reshape((10, 10), order='F'))
  print("Actual: ")
  print(labels[sel].reshape((10, 10), order='F'))
  print(np.sum(np.abs(predictions[sel] - labels[sel]) == 26), "capitalization errors")
  display_data(error_test_data[sel, :], order='C');

