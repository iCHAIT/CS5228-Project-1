from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

def weight_init(fan_in, fan_out):
	std = np.sqrt(2/fan_in)
	return (std * np.random.randn(fan_in, fan_out)).astype(np.float32)

class multi_layer_perceptron(object):
	def __init__(self, mlp_name='mlp', X_train=None, Y_train=None, X_validation=None, Y_validation=None):

		self.mlp_name = mlp_name

		self.X_train = X_train
		self.Y_train = Y_train

		self.X_validation = X_validation
		self.Y_validation = Y_validation

		self.num_train_samples = X_train.shape[0]

		self.indeces_pos = np.where(Y_train[:,1] == 1)[0]
		self.indeces_neg = np.where(Y_train[:,1] == 0)[0]

		self.num_train_samples_pos = self.indeces_pos.size
		self.num_train_samples_neg = self.indeces_neg.size

		print('Num train samples pos:%d, neg:%d' % (self.num_train_samples_pos, self.num_train_samples_neg) )

		self.batch_no_pos = 0
		self.batch_no_neg = 0

		np.random.shuffle(self.indeces_pos)
		np.random.shuffle(self.indeces_neg)

		self.batch_size = 128
		self.pos_batch_size = int(self.batch_size/2)
		self.neg_batch_size = int(self.batch_size/2)

		self.learning_rate = 0.001

		# Network Parameters
		n_input = X_train.shape[1]
		n_classes = Y_train.shape[1]
		n_hidden_1 = 16 # 1st layer number of neurons
		n_hidden_2 = 4 # 2nd layer number of neurons

		# tf Graph input
		self.X = tf.placeholder(tf.float32, [None, n_input])
		self.Y = tf.placeholder(tf.float32, [None, n_classes])
		self.lr_rate = tf.placeholder(tf.float32)

		# Store layers weight & bias
		self.weights = {
		'h_cat': tf.Variable(weight_init(47, 10)),
		'h_num': tf.Variable(weight_init(10, 10)),
		'h1': tf.Variable(weight_init(20, n_hidden_1)),
		'h2': tf.Variable(weight_init(n_hidden_1, n_hidden_2)),
		# 'h3': tf.Variable(weight_init(n_hidden_2, n_hidden_3)),
		'out': tf.Variable(weight_init(n_hidden_2, n_classes))
		}
		self.biases = {
		'b_cat': tf.Variable(np.zeros(10).astype(np.float32)),
		'b_num': tf.Variable(np.zeros(10).astype(np.float32)),
		'b1': tf.Variable(np.zeros(n_hidden_1).astype(np.float32)),
		'b2': tf.Variable(np.zeros(n_hidden_2).astype(np.float32)),
		# 'b3': tf.Variable(np.zeros(n_hidden_3).astype(np.float32)),
		'out': tf.Variable(np.zeros(n_classes).astype(np.float32))
		}

		self.layer_cat_scores = tf.add(tf.matmul(self.X[:,10:], self.weights['h_cat']), self.biases['b_cat'])
		self.layer_cat_states = tf.nn.tanh(self.layer_cat_scores)
		self.layer_num_scores = tf.add(tf.matmul(self.X[:,:10], self.weights['h_num']), self.biases['b_num'])
		self.layer_num_states = tf.nn.tanh(self.layer_num_scores)
		self.layer_1_scores = tf.add(tf.matmul(tf.concat([self.layer_num_states, self.layer_cat_states], 1), self.weights['h1']), self.biases['b1'])
		# self.layer_1_scores = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
		self.layer_1_states = tf.nn.relu(self.layer_1_scores)
		# Hidden fully connected layer
		self.layer_2_scores = tf.add(tf.matmul(self.layer_1_states, self.weights['h2']), self.biases['b2'])
		self.layer_2_states = tf.nn.relu(self.layer_2_scores)
		# Output fully connected layer with a neuron for each class
		self.logits = tf.matmul(self.layer_2_states, self.weights['out']) + self.biases['out']
		self.pred = tf.nn.softmax(self.logits)

		# Define loss and optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_rate)
		self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
		self.train_op = self.optimizer.minimize(self.loss_op)


		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

		# config = tf.ConfigProto(device_count = {'GPU': 1})
		# self.sess = tf.Session(config=config)
		self.sess = tf.Session()

		self.sess.run(self.init)


	def next_epoch(self, class_label='both'):

		if class_label == 'both':
			self.batch_no_pos = 0
			np.random.shuffle(self.indeces_pos)
			self.batch_no_neg = 0
			np.random.shuffle(self.indeces_neg)
		elif class_label == 'pos':
			self.batch_no_pos = 0
			np.random.shuffle(self.indeces_pos)
		else:
			self.batch_no_neg = 0
			np.random.shuffle(self.indeces_neg)

	def next_batch(self):

		mask = np.zeros(self.num_train_samples, dtype=bool)

		end = (self.batch_no_pos+1)*self.pos_batch_size
		if  end > self.num_train_samples_pos:
			self.next_epoch(class_label='pos')
			self.batch_no_pos = 0
			end = (self.batch_no_pos+1)*self.pos_batch_size

		start = self.batch_no_pos*self.pos_batch_size

		self.batch_no_pos += 1

		indeces_range_pos = self.indeces_pos[start:end]
		# print('Indices range:' + str(indeces_range))
		mask[indeces_range_pos]= True
		X_batch_pos = self.X_train[mask,:]
		Y_batch_pos = self.Y_train[mask,:]

		mask = np.zeros(self.num_train_samples, dtype=bool)

		end = (self.batch_no_neg+1)*self.neg_batch_size
		if end > self.num_train_samples_neg:
			self.next_epoch(class_label='neg')
			self.batch_no_neg = 0
			end = (self.batch_no_neg+1)*self.neg_batch_size

		start = self.batch_no_neg*self.neg_batch_size

		self.batch_no_neg += 1

		indeces_range_neg = self.indeces_neg[start:end]
		# print('Indices range:' + str(indeces_range))
		mask[indeces_range_neg]= True
		X_batch_neg = self.X_train[mask,:]
		Y_batch_neg = self.Y_train[mask,:]

		X_batch = np.vstack((X_batch_pos, X_batch_neg))
		Y_batch = np.vstack((Y_batch_pos, Y_batch_neg))

		return X_batch, Y_batch

	def write_to_csv_file(filepath, data_dict):

		with open(filepath,'a') as f_filepath:
			wr = csv.writer(f_filepath, dialect='excel')
			for key in data_dict.keys():
				wr.writerow(data_dict[key])

		f_filepath.close()

	def update_learning_rate(self):
		self.learning_rate = 0.95 * self.learning_rate

	def train(self):
		self.next_epoch(class_label='both')
		total_batch = int(self.num_train_samples_neg/self.neg_batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = self.next_batch()
			# Run optimization op (backprop) and cost op (to get loss value)
			_ = self.sess.run([self.train_op], feed_dict={self.X: batch_x, self.Y: batch_y, self.lr_rate: self.learning_rate})

		self.update_learning_rate()

	def predict(self, pred_type='train'):
		if pred_type == 'train':
			_loss, _pred = self.sess.run([self.loss_op, self.pred], feed_dict={self.X: self.X_train, self.Y: self.Y_train})

			_accuracy = (np.argmax(_pred,axis=1) == np.argmax(self.Y_train,axis=1)).mean()

			_mcc = matthews_corrcoef(np.argmax(self.Y_train,axis=1),np.argmax(_pred,axis=1))

		if pred_type == 'validation':
			_loss, _pred = self.sess.run([self.loss_op, self.pred], feed_dict={self.X: self.X_validation, self.Y: self.Y_validation})

			_accuracy = (np.argmax(_pred,axis=1) == np.argmax(self.Y_validation,axis=1)).mean()

			_mcc = matthews_corrcoef(np.argmax(self.Y_validation,axis=1),np.argmax(_pred,axis=1))

		return _loss, _pred, _accuracy, _mcc

	def save_model(self, data_dir):
		model_filepath = os.path.join(data_dir, self.mlp_name + '.ckpt')

		self.saver.save(self.sess, model_filepath)







