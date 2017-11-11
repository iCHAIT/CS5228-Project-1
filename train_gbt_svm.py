from __future__ import print_function

import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

def write_to_csv_file(filepath, data):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		wr.writerow(['id','prediction'])
		for i in range(data.size):
			wr.writerow([i,data[i]])

	f_filepath.close()


data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

out_file = os.path.join(data_dir, 'X_train_num_fold0.npy')
X_train = np.load(out_file).astype(np.float32)


out_file = os.path.join(data_dir, 'Y_train_num_fold0.npy')
Y_train = np.load(out_file)

out_file = os.path.join(data_dir, 'X_validation_num_fold0.npy')
X_validation = np.load(out_file).astype(np.float32)
# X_validation_num = X_validation[:,:10]
# X_validation_cat = X_validation[:,10:]

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


Y_train = np.argmax(Y_train,axis=1)
Y_validation = np.argmax(Y_validation,axis=1)

clf = SVC(class_weight='balanced', kernel='rbf', max_iter=-1, verbose=True)
clf.fit(X_train, Y_train)

train_pred = clf.predict(X_train)
train_accuracy = (train_pred == Y_train).mean()
mcc_train = matthews_corrcoef(Y_train,train_pred)

validation_pred = clf.predict(X_validation)
validation_accuracy = (validation_pred == Y_validation).mean()
mcc_validation = matthews_corrcoef(Y_validation,validation_pred)

submission_filepath = os.path.join(data_dir, 'validation_labels_gbt_svm.csv')
write_to_csv_file(filepath=submission_filepath, data=validation_pred)

print('SVM - Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
	% (train_accuracy, validation_accuracy, mcc_train,mcc_validation) )


test_labels = clf.predict(X_test)

submission_filepath = os.path.join(data_dir, 'submission_gbt_svm.csv')
write_to_csv_file(filepath=submission_filepath, data=test_labels)





