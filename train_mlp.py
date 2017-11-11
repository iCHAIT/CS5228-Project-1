from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

from mlp import multi_layer_perceptron

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

out_file = os.path.join(data_dir, 'X_train_fold0.npy')
X_train = np.load(out_file)

out_file = os.path.join(data_dir, 'Y_train_fold0.npy')
Y_train = np.load(out_file)

out_file = os.path.join(data_dir, 'X_validation_fold0.npy')
X_validation = np.load(out_file)

out_file = os.path.join(data_dir, 'Y_validation_fold0.npy')
Y_validation = np.load(out_file)


#####################

ind_neg = np.where(Y_train[:,1] == 0)[0]
ind_pos = np.where(Y_train[:,1] == 1)[0]

X_train_neg = X_train[ind_neg,:]
Y_train_neg = Y_train[ind_neg,:]

X_train_pos = X_train[ind_pos,:]
Y_train_pos = Y_train[ind_pos,:]

num_neg_samples = ind_neg.size

num_train_samples = X_train.shape[0]

indices_neg = np.arange(num_neg_samples)
np.random.shuffle(indices_neg)

num_folds = 5
num_neg_samples_in_fold = (np.floor(num_neg_samples/num_folds)).astype(int)
print('Number of negative samples in the training set of a classifier: ' + str(num_neg_samples_in_fold))

np.random.choice(5, 3, replace=False)

X_tr = list()
Y_tr = list()
for i in range(num_folds):
	mask_neg = np.zeros(num_neg_samples, dtype=bool)
	mask_neg[i*num_neg_samples_in_fold:(i+1)*num_neg_samples_in_fold]= True
	# mask_neg[np.random.choice(indices_neg, num_neg_samples_in_fold, replace=False)] = True
	print(np.sum(mask_neg))
	temp_X_neg_validation = X_train_neg[mask_neg,:]
	temp_Y_neg_validation = Y_train_neg[mask_neg,:]

	temp_X_validation = np.vstack((X_train_pos,temp_X_neg_validation))
	temp_Y_validation = np.vstack((Y_train_pos,temp_Y_neg_validation))

	X_tr.append(temp_X_validation)
	Y_tr.append(temp_Y_validation)


#####################

mlp0 = multi_layer_perceptron(mlp_name='mlp0', X_train=X_tr[0], Y_train=Y_tr[0], X_validation=X_validation, Y_validation=Y_validation)
mlp1 = multi_layer_perceptron(mlp_name='mlp1', X_train=X_tr[1], Y_train=Y_tr[1], X_validation=X_validation, Y_validation=Y_validation)
mlp2 = multi_layer_perceptron(mlp_name='mlp2', X_train=X_tr[2], Y_train=Y_tr[2], X_validation=X_validation, Y_validation=Y_validation)
mlp3 = multi_layer_perceptron(mlp_name='mlp3', X_train=X_tr[3], Y_train=Y_tr[3], X_validation=X_validation, Y_validation=Y_validation)
mlp4 = multi_layer_perceptron(mlp_name='mlp4', X_train=X_tr[4], Y_train=Y_tr[4], X_validation=X_validation, Y_validation=Y_validation)

training_epochs = 100
for epoch in range(training_epochs):
	mlp0.train()
	train_loss0, train_pred0, train_accuracy0, train_mcc0 = mlp0.predict(pred_type='train')
	validation_loss0, validation_pred0, validation_accuracy0, validation_mcc0 = mlp0.predict(pred_type='validation')
	print('MLP0 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss0, validation_loss0, train_accuracy0, validation_accuracy0, train_mcc0, validation_mcc0) )

	mlp1.train()
	train_loss1, train_pred1, train_accuracy1, train_mcc1 = mlp1.predict(pred_type='train')
	validation_loss1, validation_pred1, validation_accuracy1, validation_mcc1 = mlp1.predict(pred_type='validation')
	print('MLP1 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss1, validation_loss1, train_accuracy1, validation_accuracy1, train_mcc1, validation_mcc1) )

	mlp2.train()
	train_loss2, train_pred2, train_accuracy2, train_mcc2 = mlp2.predict(pred_type='train')
	validation_loss2, validation_pred2, validation_accuracy2, validation_mcc2 = mlp2.predict(pred_type='validation')
	print('MLP2 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss2, validation_loss2, train_accuracy2, validation_accuracy2, train_mcc2, validation_mcc2) )

	mlp3.train()
	train_loss3, train_pred3, train_accuracy3, train_mcc3 = mlp3.predict(pred_type='train')
	validation_loss3, validation_pred3, validation_accuracy3, validation_mcc3 = mlp3.predict(pred_type='validation')
	print('MLP3 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss3, validation_loss3, train_accuracy3, validation_accuracy3, train_mcc3, validation_mcc3) )

	mlp4.train()
	train_loss4, train_pred4, train_accuracy4, train_mcc4 = mlp4.predict(pred_type='train')
	validation_loss4, validation_pred4, validation_accuracy4, validation_mcc4 = mlp4.predict(pred_type='validation')
	print('MLP4 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss4, validation_loss4, train_accuracy4, validation_accuracy4, train_mcc4, validation_mcc4) )
	

	threshold = 0.41
	tmp_label0 = (validation_pred0[:,1] > threshold).astype(int)
	tmp_label1 = (validation_pred1[:,1] > threshold).astype(int)
	tmp_label2 = (validation_pred2[:,1] > threshold).astype(int)
	tmp_label3 = (validation_pred3[:,1] > threshold).astype(int)
	tmp_label4 = (validation_pred4[:,1] > threshold).astype(int)

	# label_sum = np.argmax(validation_pred0,axis=1) + np.argmax(validation_pred1,axis=1) + np.argmax(validation_pred2,axis=1) + np.argmax(validation_pred3,axis=1) + np.argmax(validation_pred4,axis=1)

	label_sum = tmp_label0 + tmp_label1 + tmp_label2 + tmp_label3 + tmp_label4
	print(label_sum)

	ensemble_label = (label_sum > 2).astype(int)

	ensemble_acc = (ensemble_label == np.argmax(Y_validation,axis=1)).mean()

	ensemble_mcc = matthews_corrcoef(ensemble_label,np.argmax(Y_validation,axis=1))

	print('ENSEMBLE - Epoch %d/%d: Acc in validation set %5.3f; MCC in validation set %5.3f' % ((epoch+1), training_epochs, ensemble_acc, ensemble_mcc) )	

# mlp0.save_model(data_dir)
# mlp1.save_model(data_dir)


	# for i in range(11):
	# 	threshold = i/10
	# 	tmp_label = validation_pred0[:,1] > threshold
	# 	tmp_acc = (tmp_label == np.argmax(Y_validation,axis=1)).mean()
	# 	tmp_mcc = matthews_corrcoef(tmp_label,np.argmax(Y_validation,axis=1))
	# 	print('THRESHOLD %5.3f: acc %5.3f, mcc %5.3f'% (threshold, tmp_acc, tmp_mcc) )




