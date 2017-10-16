from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'
model_name  = 'model_64_16_val_acc_0.877_val_loss_0.283_2017-10-05 23:09:15.ckpt'
model_filepath = os.path.join(data_dir, model_name)

out_file = os.path.join(data_dir, 'X_test.npy')
X_test = np.load(out_file)

Random_label = np.zeros((X_test.shape[0],2))

print('Test Data Shape: ' + str(X_test.shape) + '; Test Data Type: ' + str(X_test.dtype) )
print('Test Random Label Shape: ' + str(Random_label.shape) + '; Test Random Label Type: ' + str(Random_label.dtype) )

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# Network Parameters
n_input = X_test.shape[1]
n_classes = 2
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons


def write_to_csv_file(filepath, data):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		wr.writerow(['id','prediction'])
		for i in range(data.size):
			wr.writerow([i,data[i]])

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

	test_pred = sess.run([pred], feed_dict={X: X_test, Y: Random_label})
	test_pred = np.asarray(test_pred)[0,:,:]
	print(test_pred.shape)
	test_labels = np.argmax(test_pred,axis=1)
	print(np.mean(test_labels))

	submission_filepath = os.path.join(data_dir, 'submission_model_' + str(n_hidden_1) + '_' + str(n_hidden_2) + '_' 
			+ str(datetime.now()).split('.')[0] + '.csv')
	write_to_csv_file(filepath=submission_filepath, data=test_labels)



