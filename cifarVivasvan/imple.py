from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from unpickle import unpickle

data1 = unpickle("./cifar-10-batches-py/data_batch_1")
meta = unpickle('./cifar-10-batches-py/batches.meta')
test = unpickle("./cifar-10-batches-py/test_batch")

tf.logging.set_verbosity(tf.logging.INFO)


# trainData = []
# trainLabel = []
# for i in range(0,10000):
#     print(i)
#     samp = np.array(data1[b'data'][i])
#     sampr = np.reshape(samp[0:1024],(32,32))
#     sampg = np.reshape(samp[1024:2*1024],(32,32))
#     sampb = np.reshape(samp[1024*2:1024*3],(32,32))
#     trainData.append(np.dstack((sampr,sampg,sampb)))
# trainLabel = tf.constant(data1[b'labels'])
# with tf.Session() as sess:
    # print(trainData1.eval())
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print('\n----------------------------------------------------reached')
  input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      strides=(2,2),
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1,4*4*32])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  print(len(data1[b'data']))
  # Load training and eval data
  train_data = np.asarray(data1[b'data'],dtype=float)
  print("train_data",train_data.shape)
  train_labels = np.asarray(data1[b'labels'],dtype=int)
  print(len(train_labels))
#   print(train_labels)
  eval_data = np.asarray(test[b'data'],dtype=float) # Returns np.array
  eval_labels = np.asarray(test[b'labels'],dtype=int)

  # Create the Estimator
  cifar_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/cifarModelDir")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  cifar_classifier.train(
      input_fn=train_input_fn,
      steps=500,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
