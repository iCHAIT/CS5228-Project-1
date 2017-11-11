from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

out_file = os.path.join(data_dir, 'X_train_num_pca_fold0.npy')
X_train = np.load(out_file).astype(np.float32)


out_file = os.path.join(data_dir, 'Y_train_num_pca_fold0.npy')
Y_train = np.load(out_file)

out_file = os.path.join(data_dir, 'X_validation_num_pca_fold0.npy')
X_validation = np.load(out_file).astype(np.float32)
# X_validation_num_pca = X_validation[:,:10]
# X_validation_cat = X_validation[:,10:]

out_file = os.path.join(data_dir, 'Y_validation_num_pca_fold0.npy')
Y_validation = np.load(out_file)


out_file = os.path.join(data_dir, 'X_test_num_pca.npy')
X_test = np.load(out_file).astype(np.float32)


print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) + '; Data Structure: ' + str(type(X_train)) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) + '; Data Structure: ' + str(type(Y_train)))
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) + '; Data Structure: ' + str(type(X_validation)))
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) + '; Data Structure: ' + str(type(Y_validation)))

n_estimator = 30
num_numeric=0
grd = GradientBoostingClassifier(loss='exponential',n_estimators=n_estimator, min_samples_leaf=20)
grd_enc = OneHotEncoder()
grd.fit(X_train[:,num_numeric:], np.argmax(Y_train, axis=1) )
grd_enc.fit(grd.apply(X_train[:,num_numeric:])[:, :, 0])
X_train = np.hstack( (X_train[:,:num_numeric] , np.asarray(grd_enc.transform(grd.apply(X_train[:,num_numeric:])[:, :, 0]).toarray(),dtype=np.float32)))
X_validation = np.hstack( (X_validation[:,:num_numeric], np.asarray(grd_enc.transform(grd.apply(X_validation[:,num_numeric:])[:, :, 0]).toarray(),dtype=np.float32)))

print(np.amax(np.sum(X_validation,axis=1)))

X_test = np.hstack( (X_test[:,:num_numeric] , np.asarray(grd_enc.transform(grd.apply(X_test[:,num_numeric:])[:, :, 0]).toarray(),dtype=np.float32)))

print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) + '; Data Structure: ' + str(type(X_train)) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) + '; Data Structure: ' + str(type(Y_train)))
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) + '; Data Structure: ' + str(type(X_validation)))
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) + '; Data Structure: ' + str(type(Y_validation)))



num_train_samples = X_train.shape[0]

indeces_pos = np.where(Y_train[:,1] == 1)[0]
indeces_neg = np.where(Y_train[:,1] == 0)[0]

num_train_samples_pos = indeces_pos.size
num_train_samples_neg = indeces_neg.size

print(num_train_samples_pos)
print(num_train_samples_neg)

batch_no_pos = 0
batch_no_neg = 0

np.random.shuffle(indeces_pos)
np.random.shuffle(indeces_neg)


def next_epoch(class_label='both'):
	global batch_no_pos
	global batch_no_neg

	if class_label == 'both':
		batch_no_pos = 0
		np.random.shuffle(indeces_pos)
		batch_no_neg = 0
		np.random.shuffle(indeces_neg)
	elif class_label == 'pos':
		batch_no_pos = 0
		np.random.shuffle(indeces_pos)
	else:
		batch_no_neg = 0
		np.random.shuffle(indeces_neg)
	# print('############ Next Epoch ############')

def next_batch():
	global batch_no_pos
	global batch_no_neg

	mask = np.zeros(num_train_samples, dtype=bool)
	
	end = (batch_no_pos+1)*pos_batch_size
	if  end > num_train_samples_pos:
		next_epoch(class_label='pos')
		batch_no_pos = 0
		end = (batch_no_pos+1)*pos_batch_size

	start = batch_no_pos*pos_batch_size

	batch_no_pos += 1
	
	indeces_range_pos = indeces_pos[start:end]
	# print('Indices range:' + str(indeces_range))
	mask[indeces_range_pos]= True
	X_batch_pos = X_train[mask,:]
	Y_batch_pos = Y_train[mask,:]

	mask = np.zeros(num_train_samples, dtype=bool)
	
	end = (batch_no_neg+1)*neg_batch_size
	if  end > num_train_samples_neg:
		next_epoch(class_label='neg')
		batch_no_neg = 0
		end = (batch_no_neg+1)*neg_batch_size

	start = batch_no_neg*neg_batch_size

	batch_no_neg += 1
	
	indeces_range_neg = indeces_neg[start:end]
	# print('Indices range:' + str(indeces_range))
	mask[indeces_range_neg]= True
	X_batch_neg = X_train[mask,:]
	Y_batch_neg = Y_train[mask,:]

	X_batch = np.vstack((X_batch_pos, X_batch_neg))
	Y_batch = np.vstack((Y_batch_pos, Y_batch_neg))

	return X_batch, Y_batch

def write_to_csv_file(filepath, data_dict):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		for key in data_dict.keys():
			wr.writerow(data_dict[key])

	f_filepath.close()


def write_to_csv_file2(filepath, data):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		wr.writerow(['id','prediction'])
		for i in range(data.size):
			wr.writerow([i,data[i]])

	f_filepath.close()

def weight_init(fan_in, fan_out):
		std = np.sqrt(2/fan_in)
		return (std * np.random.randn(fan_in, fan_out)).astype(np.float32)

def update_learning_rate():
	global learning_rate
	learning_rate = 0.95 * learning_rate

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 256
half_batch_size = int(batch_size/2)
pos_batch_size = int(batch_size/4)
neg_batch_size = 3*int(batch_size/4)
display_step = 1

# Network Parameters
n_input = X_train.shape[1]
n_classes = Y_train.shape[1]
n_hidden_1 = 16 # 1st layer number of neurons
n_hidden_2 = 4 # 2nd layer number of neurons

# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
lr_rate = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(weight_init(n_input, n_hidden_1)),
    'h2': tf.Variable(weight_init(n_hidden_1, n_hidden_2)),
    'out': tf.Variable(weight_init(n_hidden_2, n_classes))
}
biases = {
    'b1': tf.Variable(np.zeros(n_hidden_1).astype(np.float32)),
    'b2': tf.Variable(np.zeros(n_hidden_2).astype(np.float32)),
    'out': tf.Variable(np.zeros(n_classes).astype(np.float32))
}



# Hidden fully connected layer with 256 neurons
print('X shape:' + str(X.shape) + ': X type:' + str(X.dtype))
print('Weights shape:' + str(weights['h1'].shape) + ': Weights type:' + str(weights['h1'].dtype))
layer_1_scores = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
layer_1_states = tf.nn.relu(layer_1_scores)
# Hidden fully connected layer with 256 neurons
layer_2_scores = tf.add(tf.matmul(layer_1_states, weights['h2']), biases['b2'])
layer_2_states = tf.nn.relu(layer_2_scores)
# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2_states, weights['out']) + biases['out']
pred = tf.nn.softmax(logits)

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = optimizer.minimize(loss_op)

train_loss_history = []
validation_loss_history = []
train_acc_history = []
validation_acc_history = []
max_val_acc = 0
min_val_loss = 10000
max_val_mcc = 0

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		next_epoch(class_label='both')
		avg_cost = 0.
		total_batch = int(num_train_samples_neg/neg_batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = next_batch()
			# Run optimization op (backprop) and cost op (to get loss value)
			_ = sess.run([train_op], feed_dict={X: batch_x, Y: batch_y, lr_rate: learning_rate})

		# Display logs per epoch step
		if epoch % display_step == 0:
			train_loss, train_pred = sess.run([loss_op, pred], feed_dict={X: X_train, Y: Y_train})
			train_loss_history.append(train_loss)
			
			thrshld = 0.5
			train_pred_label = (train_pred[:,1] > thrshld).astype(int)
			train_accuracy = (train_pred_label == np.argmax(Y_train,axis=1)).mean()
			train_acc_history.append(train_accuracy)

			mcc_train = matthews_corrcoef(np.argmax(Y_train,axis=1),train_pred_label)

			validation_loss, validation_pred = sess.run([loss_op, pred], feed_dict={X: X_validation, Y: Y_validation})
			validation_loss_history.append(validation_loss)

			validation_pred_label = (validation_pred[:,1] > thrshld).astype(int)
			validation_accuracy = (validation_pred_label == np.argmax(Y_validation,axis=1)).mean()
			validation_acc_history.append(validation_accuracy)

			mcc_validation = matthews_corrcoef(np.argmax(Y_validation,axis=1),validation_pred_label)
			conf_validation = confusion_matrix(np.argmax(Y_validation,axis=1),validation_pred_label)


			print('Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss, validation_loss, train_accuracy, validation_accuracy, mcc_train,mcc_validation) )

			print(conf_validation)

			# if (round(validation_accuracy,3) > round(max_val_acc,3)) and (round(validation_loss,3) < round(min_val_loss,3)):
			if (round(validation_loss,3) < round(min_val_loss,3)):
				max_val_acc = validation_accuracy
				min_val_loss = validation_loss

				model_filepath = os.path.join(data_dir, 'best_model_gbt_nn_pca' + '.ckpt')

				saver.save(sess, model_filepath)

				csv_filename = os.path.join(data_dir, 'gbt_nn_pca_validation_pred' + '.txt')

				np.savetxt(csv_filename, validation_pred, delimiter=',')


			
			# if (round(mcc_validation,3) > round(max_val_mcc,3)):
			# 	max_val_mcc = mcc_validation
				
			# 	model_filepath = os.path.join(data_dir, 'model_' + str(n_hidden_1) + '_' + str(n_hidden_2) + '_val_mcc_' 
			# 		+ str(round(mcc_validation,3))  + '_' + str(datetime.now()).split('.')[0] + '.ckpt')
			# 	saver.save(sess, model_filepath)

		# Update learning rate
		# if epoch % 20 == 0:
		update_learning_rate()


	stats = { 'train_loss_history': train_loss_history,
	'validation_loss_history': validation_loss_history,
	'train_acc_history': train_acc_history,
	'validation_acc_history': validation_acc_history}

	stats_filepath = os.path.join(data_dir, 'gbt_nn_pca_loss_acc_history.csv')
	write_to_csv_file(filepath=stats_filepath, data_dict=stats)

	model_filepath = os.path.join(data_dir, 'best_model_gbt_nn_pca' + '.ckpt')
	saver.restore(sess, model_filepath)
	
	Random_label = np.zeros((X_test.shape[0],2))

	test_pred = sess.run([pred], feed_dict={X: X_test, Y: Random_label})
	test_pred = np.asarray(test_pred)[0,:,:]
	print(test_pred.shape)
	test_labels = np.argmax(test_pred,axis=1)
	print(np.mean(test_labels))

	submission_filepath = os.path.join(data_dir, 'submission_gbt_nn_pca.csv')
	write_to_csv_file2(filepath=submission_filepath, data=test_labels)




