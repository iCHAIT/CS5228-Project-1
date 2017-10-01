from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'
model_name  = 'model_128_64_val_acc_0.912_val_loss_0.207_2017-10-01 20:23:19.ckpt'
model_filepath = os.path.join(data_dir, model_name)

out_file = os.path.join(data_dir, 'X_validation_fold0.npy')
X_validation = np.load(out_file)

out_file = os.path.join(data_dir, 'Y_validation_fold0.npy')
Y_validation = np.load(out_file)

print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) )
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) )

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# Network Parameters
n_input = X_validation.shape[1]
n_classes = 2
n_hidden_1 = 128 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons


def write_to_csv_file(filepath, data_dict):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		for key in data_dict.keys():
			wr.writerow(data_dict[key])

	f_filepath.close()

def weight_init(fan_in, fan_out):
		std = np.sqrt(2/fan_in)
		return (std * np.random.randn(fan_in, fan_out)).astype(np.float64)

# tf Graph input
X = tf.placeholder(tf.float64, [None, n_input])
Y = tf.placeholder(tf.float64, [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(weight_init(n_input, n_hidden_1)),
    'h2': tf.Variable(weight_init(n_hidden_1, n_hidden_2)),
    'out': tf.Variable(weight_init(n_hidden_2, n_classes))
}
biases = {
    'b1': tf.Variable(np.zeros(n_hidden_1).astype(np.float64)),
    'b2': tf.Variable(np.zeros(n_hidden_2).astype(np.float64)),
    'out': tf.Variable(np.zeros(n_classes).astype(np.float64))
}


# Hidden fully connected layer
layer_1_scores = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
layer_1_states = tf.nn.relu(layer_1_scores)
# Hidden fully connected layer with
layer_2_scores = tf.add(tf.matmul(layer_1_states, weights['h2']), biases['b2'])
layer_2_states = tf.nn.relu(layer_2_scores)
# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2_states, weights['out']) + biases['out']
pred = tf.nn.softmax(logits)

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = optimizer.minimize(loss_op)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, model_filepath)

	validation_loss, validation_pred = sess.run([loss_op, pred], feed_dict={X: X_validation, Y: Y_validation})
	validation_accuracy = (np.argmax(validation_pred,axis=1) == np.argmax(Y_validation,axis=1)).mean()


	mcc_validation = matthews_corrcoef(np.argmax(Y_validation,axis=1),np.argmax(validation_pred,axis=1))
	print('Mathews Correlation Coefficient:' + str(mcc_validation))

	print('loss in validation set ' + str(validation_loss) + '; Acc in validation set ' + str(validation_accuracy) )



