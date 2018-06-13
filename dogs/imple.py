from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile,join

tf.logging.set_verbosity(tf.logging.INFO)
traindirnames = ['./beagleSet',"./bulldogSet","./greatdaneSet"]
evaldirnames = ['./beagleESet',"./bulldogESet","./greatdaneESet"]
def cnn_model_fn(features, labels, mode):
    print("reached")
	# Input Layer
    input_layer = features['x']
    # Convolutuion Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        padding="same",
        strides=(4,4),
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    norm1 = tf.nn.lrn(pool1)

    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=256,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2)

    norm2 = tf.nn.lrn(pool2)

    conv3 = tf.layers.conv2d(
        inputs=norm2,
        filters=384,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu
    )

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=384,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu
    )

    finalpool = tf.layers.max_pooling2d(inputs=conv5,pool_size=[3,3],strides=2)

    flatpool = tf.reshape(finalpool,[-1,finalpool.shape[1]*finalpool.shape[2]*finalpool.shape[3]])  
    print(finalpool.shape[3])
    dense = tf.layers.dense(inputs=flatpool, units=4096, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=4096,activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2,units=1000)
    
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
        print("123")
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    # with tf.Session() as sess:
        # print(len(sess.run(finalpool)))

    return input_layer

train_data = []
train_labels = []
eval_data = []
eval_labels = []
def main(unused_arg):
    for curDir in traindirnames:
        allfiles = [f for f in listdir(curDir) if isfile(join(curDir,f))]
        for i in allfiles:
            train_data.append(cv2.imread(curDir+"/"+i))
        train_labels.extend([traindirnames.index(curDir)]*len(allfiles))
    
    for curDir in evaldirnames:
        allfiles = [f for f in listdir(curDir) if isfile(join(curDir,f))]
        for i in allfiles:
            eval_data.append(cv2.imread(curDir+"/"+i))
        eval_labels.extend([evaldirnames.index(curDir)]*len(allfiles))
    
     # Create the Estimator
    alexnet = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./Desktop/coding/machineLearning/tensorFlow/dogs")
    #train_labels.extend([0,1,2])

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    # print(tf.placeholder(train_data).shape)
    # print("asdf"+5)
    # print(train_data[0])
    # print(td1.shape)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x" : np.asarray(train_data,dtype=np.float16)},
        y= np.asarray(train_labels,dtype=int),
        batch_size=2,
        num_epochs=None,
        shuffle=True)

    alexnet.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

	# Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False)
    eval_results = alexnet.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
