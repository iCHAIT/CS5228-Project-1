from __future__ import print_function

import numpy as np
import os
from datetime import datetime
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def write_to_csv_file(filepath, data):

	with open(filepath,'a') as f_filepath:
		wr = csv.writer(f_filepath, dialect='excel')
		wr.writerow(['id','prediction'])
		for i in range(data.size):
			wr.writerow([i,data[i]])

	f_filepath.close()

def read_from_csv_file(filepath):
	data = list()
	with open(filepath,'r') as f_filepath:
		rd = csv.reader(f_filepath, dialect='excel')
		next(rd, None)
		for row in rd:
			data.append(row[1])

	f_filepath.close()

	return np.asarray(data)[:,np.newaxis]


data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'


out_file = os.path.join(data_dir, 'validation_labels_svm.csv')
X1 = read_from_csv_file(out_file)

out_file = os.path.join(data_dir, 'validation_labels_random_forest.csv')
X2 = read_from_csv_file(out_file)

out_file = os.path.join(data_dir, 'validation_labels_decision_tree.csv')
X3 = read_from_csv_file(out_file)

out_file = os.path.join(data_dir, 'validation_labels_nn3.csv')
X4 = read_from_csv_file(out_file)
print(X1.shape)
print(X2.shape)
print(X3.shape)
print(X4.shape)

X = np.hstack((X1,X2,X3,X4))

print(X.shape)

out_file = os.path.join(data_dir, 'Y_validation_fold0.npy')
Y = np.load(out_file)
Y = np.argmax(Y,axis=1)

out_file = os.path.join(data_dir, 'X_test.npy')
X_test = np.load(out_file)

num_train_samples = X.shape[0]
indeces = np.arange(num_train_samples)
np.random.shuffle(indeces)
num_samples_in_val = (np.floor(num_train_samples/5)).astype(int)

mask = np.zeros(num_train_samples, dtype=bool)
indeces_range = indeces[:num_samples_in_val]
mask[indeces_range]= True
X_validation = X[mask,:]
Y_validation = Y[mask]
mask = np.invert(mask)
X_train = X[mask,:]
Y_train = Y[mask]


print('Training Data Shape: ' + str(X_train.shape) + '; Training Data Type: ' + str(X_train.dtype) )
print('Training Label Shape: ' + str(Y_train.shape) + '; Training Label Type: ' + str(Y_train.dtype) )
print('Validation Data Shape: ' + str(X_validation.shape) + '; Validation Data Type: ' + str(X_validation.dtype) )
print('Validation Label Shape: ' + str(Y_validation.shape) + '; Validation Label Type: ' + str(Y_validation.dtype) )


# clf = SVC(class_weight='balanced', kernel='rbf', max_iter=-1, verbose=True)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, Y_train)

train_pred = clf.predict(X_train)
train_accuracy = (train_pred == Y_train).mean()
mcc_train = matthews_corrcoef(Y_train,train_pred)

validation_pred = clf.predict(X_validation)
validation_accuracy = (validation_pred == Y_validation).mean()
mcc_validation = matthews_corrcoef(Y_validation,validation_pred)

# submission_filepath = os.path.join(data_dir, 'validation_labels_svm.csv')
# write_to_csv_file(filepath=submission_filepath, data=validation_pred)

print('SVM - Acc in training set %5.3f, in validation set %5.3f; MCC in training set %5.3f, in validation set %5.3f'
	% (train_accuracy, validation_accuracy, mcc_train,mcc_validation) )


# test_labels = clf.predict(X_test)

# submission_filepath = os.path.join(data_dir, 'test_labels_ensemble_svm.csv')
# write_to_csv_file(filepath=submission_filepath, data=test_labels)



