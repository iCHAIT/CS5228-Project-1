from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder

from mlp_cascade import multi_layer_perceptron

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

out_file = os.path.join(data_dir, 'X_train_num_fold0.npy')
X_train = np.load(out_file).astype(np.float32)

out_file = os.path.join(data_dir, 'Y_train_num_fold0.npy')
Y_train = np.load(out_file)

out_file = os.path.join(data_dir, 'X_validation_num_fold0.npy')
X_validation = np.load(out_file).astype(np.float32)

out_file = os.path.join(data_dir, 'Y_validation_num_fold0.npy')
Y_validation = np.load(out_file)


out_file = os.path.join(data_dir, 'X_test_num.npy')
X_test = np.load(out_file).astype(np.float32)

print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) + '; Data Structure: ' + str(type(X_train)) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) + '; Data Structure: ' + str(type(Y_train)))
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) + '; Data Structure: ' + str(type(X_validation)))
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) + '; Data Structure: ' + str(type(Y_validation)))

n_estimator = 10
grd = GradientBoostingClassifier(loss='exponential',n_estimators=n_estimator, min_samples_leaf=20)
grd_enc = OneHotEncoder()
grd.fit(X_train[:,10:], np.argmax(Y_train, axis=1) )
grd_enc.fit(grd.apply(X_train[:,10:])[:, :, 0])
X_train = np.hstack( (X_train[:,:10] , np.asarray(grd_enc.transform(grd.apply(X_train[:,10:])[:, :, 0]).toarray(),dtype=np.float32)))
X_validation = np.hstack( (X_validation[:,:10], np.asarray(grd_enc.transform(grd.apply(X_validation[:,10:])[:, :, 0]).toarray(),dtype=np.float32)))

print(np.amax(np.sum(X_validation,axis=1)))

X_test = np.hstack( (X_test[:,:10] , np.asarray(grd_enc.transform(grd.apply(X_test[:,10:])[:, :, 0]).toarray(),dtype=np.float32)))

print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) + '; Data Structure: ' + str(type(X_train)) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) + '; Data Structure: ' + str(type(Y_train)))
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) + '; Data Structure: ' + str(type(X_validation)))
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) + '; Data Structure: ' + str(type(Y_validation)))


mlp0 = multi_layer_perceptron(mlp_name='mlp0', X_train=X_train, Y_train=Y_train, X_validation=X_validation, Y_validation=Y_validation)

training_epochs = 100
for epoch in range(training_epochs):
	mlp0.train()
	train_loss0, train_pred0, train_accuracy0, train_mcc0 = mlp0.predict(pred_type='train')
	validation_loss0, validation_pred0, validation_accuracy0, validation_mcc0 = mlp0.predict(pred_type='validation')
	print('MLP0 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss0, validation_loss0, train_accuracy0, validation_accuracy0, train_mcc0, validation_mcc0) )


validation_loss0, validation_pred0, validation_accuracy0, validation_mcc0 = mlp0.predict(pred_type='validation')
train_loss0, train_pred0, train_accuracy0, train_mcc0 = mlp0.predict(pred_type='train')

indeces = np.where( train_pred0[:,1]>train_pred0[:,0] )[0]

X_train_new = X_train[indeces,:]
Y_train_new = Y_train[indeces,:]

mlp1 = multi_layer_perceptron(mlp_name='mlp1', X_train=X_train_new, Y_train=Y_train_new, X_validation=X_validation, Y_validation=Y_validation)

training_epochs = 100
for epoch in range(training_epochs):
	mlp1.train()
	train_loss1, train_pred1, train_accuracy1, train_mcc1 = mlp1.predict(pred_type='train')
	validation_loss1, validation_pred1, validation_accuracy1, validation_mcc1 = mlp1.predict(pred_type='validation')
	print('MLP1 - Epoch %d/%d: loss in training set %5.3f, validation set %5.3f; Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
			 % ((epoch+1), training_epochs, train_loss1, validation_loss1, train_accuracy1, validation_accuracy1, train_mcc1, validation_mcc1) )



validation_loss1, validation_pred1, validation_accuracy1, validation_mcc1 = mlp1.predict(pred_type='validation')


indeces_new = np.where( validation_pred0[:,1]>validation_pred0[:,0] )[0]
cascade_pred = validation_pred0
cascade_pred[indeces_new,:] = validation_pred1[indeces_new,:]


cascade_label = np.argmax(cascade_pred,axis=1)

cascade_acc = (cascade_label == np.argmax(Y_validation,axis=1)).mean()

cascade_mcc = matthews_corrcoef(cascade_label,np.argmax(Y_validation,axis=1))

print('Cascade - Acc in validation set %5.3f; MCC in validation set %5.3f' % (cascade_acc, cascade_mcc) )	








