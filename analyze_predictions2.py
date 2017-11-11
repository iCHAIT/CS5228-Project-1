from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

csv_filename = os.path.join(data_dir, 'Y_validation_fold0' + '.txt')
Y_validation = np.loadtxt(csv_filename, delimiter=',')
Y_val = np.argmax(Y_validation,axis=1)


csv_filenames = ['validation_labels_gbt_nn.csv',
'validation_labels_gbt_svm.csv',
'validation_labels_gbt_decision_tree.csv',
'validation_labels_gbt_logistic_regression.csv']

label = list()
for csv_filename in csv_filenames:
	validation_pred = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
	tmp_label=validation_pred[:,1]
	mcc = matthews_corrcoef(Y_val,tmp_label)
	conf = confusion_matrix(Y_val,tmp_label)
	acc = accuracy_score(Y_val,tmp_label)

	print(csv_filename)
	print('ACC: %5.3f' % acc)
	print('MCC: %5.3f' % mcc)
	print('CONFUSION MATRIX:')
	print(conf)

	label.append(tmp_label)

label_arr = np.asarray(label).T
# print(label_arr.shape)

clf = DecisionTreeClassifier(max_depth=1)
clf.fit(label_arr, Y_val)
train_pred = clf.predict(label_arr)

mcc = matthews_corrcoef(Y_val,train_pred)
conf = confusion_matrix(Y_val,train_pred)
acc = accuracy_score(Y_val,train_pred)

print('ENSEMBLE RESULTS')
print('ACC: %5.3f' % acc)
print('MCC: %5.3f' % mcc)
print('CONFUSION MATRIX:')
print(conf)


