from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
# import tensorflow as tf
import csv
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

csv_filename = os.path.join(data_dir, 'Y_validation_num_fold0' + '.txt')
Y_validation = np.loadtxt(csv_filename, delimiter=',')

csv_filename = os.path.join(data_dir, 'gbt_nn_validation_pred' + '.txt')
validation_pred = np.loadtxt(csv_filename, delimiter=',')

n_classes = Y_validation.shape[1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_validation[:, i], validation_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
plt.subplot(2, 1, 1)
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")


mcc = list()
threshold = list()
for i in range(101):
	thrshld = i/100
	threshold.append(thrshld)
	tmp_label = (validation_pred[:,1] > thrshld).astype(int)
	mcc.append(matthews_corrcoef(tmp_label,np.argmax(Y_validation,axis=1)))

max_mcc = np.amax(mcc)
thrshld_at_max_mcc = threshold[np.argmax(mcc)]

plt.subplot(2, 1, 2)
plt.plot(threshold, mcc, color='darkorange', lw=lw, label='MCC (max = %0.2f @ thrshld = %0.2f)' % (max_mcc,thrshld_at_max_mcc) )
plt.scatter(thrshld_at_max_mcc, max_mcc, color='navy')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('MCC Value')
plt.title('MCC for Different Threshold Values')
plt.legend(loc="lower right")

plt.tight_layout(pad=1.0)

plt.show()
# plt.hist(nn3_validation_pred, 2, normed=1, facecolor='g', alpha=0.75)

# Plot the loss function and train / test accuracies
# fig = plt.figure()
# plt.subplot(2, 1, 1)
# # plt.scatter(Y_validation[:,0], label='True Labels')
# plt.scatter(Y_validation[:,0],nn3_validation_pred[:,0], label='Predictions')
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
