from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'
model_name  = 'best_model_nn3_2branch.ckpt'
model_filepath = os.path.join(data_dir, model_name)

out_file = os.path.join(data_dir, 'X_test.npy')
# out_file = os.path.join(data_dir, 'X_validation_fold0.npy')
X_test = np.load(out_file)

Random_label = np.zeros((X_test.shape[0],2))

print('Test Data Shape: ' + str(X_test.shape) + '; Test Data Type: ' + str(X_test.dtype) )
print('Test Random Label Shape: ' + str(Random_label.shape) + '; Test Random Label Type: ' + str(Random_label.dtype) )


def write_to_csv_file(filepath, data):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		wr.writerow(['id','prediction'])
		for i in range(data.size):
			wr.writerow([i,data[i]])

	f_filepath.close()

def weight_init(fan_in, fan_out):
		std = np.sqrt(2/fan_in)
		return (std * np.random.randn(fan_in, fan_out)).astype(np.float32)

# Network Parameters
n_input = X_test.shape[1]
n_classes = 2
n_hidden_1 = 16 # 1st layer number of neurons
n_hidden_2 = 4 # 2nd layer number of neurons
# n_hidden_3 = 2 # 3rd layer number of neurons

# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
lr_rate = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
	'h_cat': tf.Variable(weight_init(47, 10)),
	'h_num': tf.Variable(weight_init(10, 10)),
    'h1': tf.Variable(weight_init(20, n_hidden_1)),
    'h2': tf.Variable(weight_init(n_hidden_1, n_hidden_2)),
    # 'h3': tf.Variable(weight_init(n_hidden_2, n_hidden_3)),
    'out': tf.Variable(weight_init(n_hidden_2, n_classes))
}
biases = {
	'b_cat': tf.Variable(np.zeros(10).astype(np.float32)),
	'b_num': tf.Variable(np.zeros(10).astype(np.float32)),
    'b1': tf.Variable(np.zeros(n_hidden_1).astype(np.float32)),
    'b2': tf.Variable(np.zeros(n_hidden_2).astype(np.float32)),
    # 'b3': tf.Variable(np.zeros(n_hidden_3).astype(np.float32)),
    'out': tf.Variable(np.zeros(n_classes).astype(np.float32))
}



# Hidden fully connected layer with 256 neurons
print('X shape:' + str(X.shape) + ': X type:' + str(X.dtype))
print('Weights shape:' + str(weights['h1'].shape) + ': Weights type:' + str(weights['h1'].dtype))
layer_cat_scores = tf.add(tf.matmul(X[:,10:], weights['h_cat']), biases['b_cat'])
layer_cat_states = tf.nn.tanh(layer_cat_scores)
layer_num_scores = tf.add(tf.matmul(X[:,:10], weights['h_num']), biases['b_num'])
layer_num_states = tf.nn.tanh(layer_num_scores)
layer_1_scores = tf.add(tf.matmul(tf.concat([layer_num_states, layer_cat_states], 1), weights['h1']), biases['b1'])
layer_1_states = tf.nn.relu(layer_1_scores)
# Hidden fully connected layer with 256 neurons
layer_2_scores = tf.add(tf.matmul(layer_1_states, weights['h2']), biases['b2'])
layer_2_states = tf.nn.relu(layer_2_scores)
# Hidden fully connected layer with 256 neurons
# layer_3_scores = tf.add(tf.matmul(layer_2_states, weights['h3']), biases['b3'])
# layer_3_states = tf.nn.relu(layer_3_scores)
# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2_states, weights['out']) + biases['out']
pred = tf.nn.softmax(logits)

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = optimizer.minimize(loss_op)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, model_filepath)

	test_pred = sess.run([pred], feed_dict={X: X_test, Y: Random_label})
	test_pred = np.asarray(test_pred)[0,:,:]
	print(test_pred.shape)
	test_labels = np.argmax(test_pred,axis=1)
	print(np.mean(test_labels))

	submission_filepath = os.path.join(data_dir, 'submission.csv')
	# submission_filepath = os.path.join(data_dir, 'validation_labels_nn3.csv')

	write_to_csv_file(filepath=submission_filepath, data=test_labels)



