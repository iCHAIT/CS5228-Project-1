from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

out_file = os.path.join(data_dir, 'X_train_fold0.npy')
X_train = np.load(out_file)

out_file = os.path.join(data_dir, 'Y_train_fold0.npy')
Y_train = np.load(out_file)

out_file = os.path.join(data_dir, 'X_validation_fold0.npy')
X_validation = np.load(out_file)

out_file = os.path.join(data_dir, 'Y_validation_fold0.npy')
Y_validation = np.load(out_file)

print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) )
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) )
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) )

num_train_samples = X_train.shape[0]
batch_no = 0
indeces = np.arange(num_train_samples)
np.random.shuffle(indeces)

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# Network Parameters
n_input = X_train.shape[1]
n_classes = Y_train.shape[1]
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons

def next_epoch():
	global batch_no
	batch_no = 0
	np.random.shuffle(indeces)
	# print('############ Next Epoch ############')

def next_batch():
	global batch_no
	mask = np.zeros(num_train_samples, dtype=bool)
	
	end = (batch_no+1)*batch_size
	if  end > num_train_samples:
		batch_no = 0
		end = (batch_no+1)*batch_size

	start = batch_no*batch_size

	batch_no += 1
	
	indeces_range = indeces[start:end]
	# print('Indices range:' + str(indeces_range))
	mask[indeces_range]= True
	X_batch = X_train[mask,:]
	Y_batch = Y_train[mask,:]

	return X_batch, Y_batch

def write_to_csv_file(filepath, data_dict):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		for key in data_dict.keys():
			wr.writerow(data_dict[key])

	f_filepath.close()

# for epoch in range(training_epochs):
# 	next_epoch()
# 	avg_cost = 0.
# 	total_batch = int(num_train_samples/batch_size)
# 	# Loop over all batches
# 	for i in range(total_batch):
# 		batch_x, batch_y = next_batch()
# 	# Display logs per epoch step
# 	if epoch % display_step == 0:
# 		print('Epoch:', '%04d' % (epoch+1), 'cost={:.9f}'.format(avg_cost))
# print('Optimization Finished!')

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
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = optimizer.minimize(loss_op)

train_loss_history = []
validation_loss_history = []
train_acc_history = []
validation_acc_history = []
max_val_acc = 0
min_val_loss = 10000

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		next_epoch()
		avg_cost = 0.
		total_batch = int(num_train_samples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = next_batch()
			# Run optimization op (backprop) and cost op (to get loss value)
			_ = sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})

		# Display logs per epoch step
		if epoch % display_step == 0:
			train_loss, train_pred = sess.run([loss_op, pred], feed_dict={X: X_train, Y: Y_train})
			train_loss_history.append(train_loss)
			
			train_accuracy = (np.argmax(train_pred,axis=1) == np.argmax(Y_train,axis=1)).mean()
			train_acc_history.append(train_accuracy)

			mcc_train = matthews_corrcoef(np.argmax(Y_train,axis=1),np.argmax(train_pred,axis=1))

			validation_loss, validation_pred = sess.run([loss_op, pred], feed_dict={X: X_validation, Y: Y_validation})
			validation_loss_history.append(validation_loss)

			validation_accuracy = (np.argmax(validation_pred,axis=1) == np.argmax(Y_validation,axis=1)).mean()
			validation_acc_history.append(validation_accuracy)

			mcc_validation = matthews_corrcoef(np.argmax(Y_validation,axis=1),np.argmax(validation_pred,axis=1))

			print('Epoch ' + str((epoch+1)) + '/' + str(training_epochs) + ': loss in training set ' + str(train_loss)
				+ ', validation set ' + str(validation_loss) + '; Acc in training set ' + str(train_accuracy)
				+ ', in validation set ' + str(validation_accuracy) + '; MCC in training set ' + str(mcc_train)
				+ ', in validation set ' + str(mcc_validation) )

			if (round(validation_accuracy,3) > round(max_val_acc,3)) and (round(validation_loss,3) < round(min_val_loss,3)):
				max_val_acc = validation_accuracy
				min_val_loss = validation_loss

				model_filepath = os.path.join(data_dir, 'model_' + str(n_hidden_1) + '_' + str(n_hidden_2) + '_val_acc_' 
					+ str(round(validation_accuracy,3)) + '_val_loss_' + str(round(validation_loss,3))  + '_' 
					+ str(datetime.now()).split('.')[0] + '.ckpt')
				saver.save(sess, model_filepath)
				
	stats = { 'train_loss_history': train_loss_history,
	'validation_loss_history': validation_loss_history,
	'train_acc_history': train_acc_history,
	'validation_acc_history': validation_acc_history}

	stats_filepath = os.path.join(data_dir, 'model_' + str(n_hidden_1) + '_' + str(n_hidden_2) + '_loss_acc_history_' 
		+ str(datetime.now()).split('.')[0] + '.csv')
	write_to_csv_file(filepath=stats_filepath, data_dict=stats)




	# Plot the loss function and train / test accuracies
	# fig = plt.figure()
	# plt.subplot(2, 1, 1)
	# plt.plot(train_loss_history, label='train')
	# plt.plot(validation_loss_history, label='validation')
	# plt.title('Loss history')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.grid(b=True, linestyle='-.')

	# plt.subplot(2, 1, 2)
	# plt.plot(train_acc_history, label='train')
	# plt.plot(validation_acc_history, label='validation')
	# plt.title('Classification accuracy history')
	# plt.xlabel('Epoch')
	# plt.ylabel('Clasification accuracy')
	# plt.legend()
	# plt.grid(b=True, linestyle='-.')

	# plt.tight_layout(pad=1.0)

	# fig.savefig( ('loss_vs_epoch_and_acc_vs_epoch_nn1_' + str(datetime.now()).split('.')[0] + '.png') , bbox_inches = 'tight')

	# plt.show()



