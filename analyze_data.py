import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed','y']

features_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
'day_of_week', 'poutcome']

data_file = os.path.join(data_dir, 'train.csv')
train_data = pd.read_csv(data_file, usecols=features)
num_train = len(train_data)


data = train_data


age_arr = np.asarray(data['age'])
age_arr = (( (age_arr - age_arr.min()) / (age_arr.max() - age_arr.min()) )*2 - 1)[:,np.newaxis]

duration_arr = np.asarray(data['duration'])
duration_arr = (( (duration_arr - duration_arr.min()) / (duration_arr.max() - duration_arr.min()) )*2 - 1)[:,np.newaxis]

campaign_arr = np.asarray(data['campaign'])
campaign_arr = (( (campaign_arr - campaign_arr.min()) / (campaign_arr.max() - campaign_arr.min()) )*2 - 1)[:,np.newaxis]

pdays_arr = np.asarray(data['pdays'])
ind = np.asarray(pdays_arr == 999)
pdays_mean = np.mean(pdays_arr[np.invert(ind)])
pdays_arr[ind] = pdays_mean
pdays_arr = (( (pdays_arr - pdays_arr.min()) / (pdays_arr.max() - pdays_arr.min()) )*2 - 1)[:,np.newaxis]

previous_arr = np.asarray(data['previous'])
previous_arr = (( (previous_arr - previous_arr.min()) / (previous_arr.max() - previous_arr.min()) )*2 - 1)[:,np.newaxis]

emp_var_rate_arr = np.asarray(data['emp.var.rate'])
emp_var_rate_arr = (( (emp_var_rate_arr - emp_var_rate_arr.min()) / (emp_var_rate_arr.max() - emp_var_rate_arr.min()) )*2 - 1)[:,np.newaxis]

cons_price_idx_arr = np.asarray(data['cons.price.idx'])
cons_price_idx_arr = (( (cons_price_idx_arr - cons_price_idx_arr.min()) / (cons_price_idx_arr.max() - cons_price_idx_arr.min()) )*2 - 1)[:,np.newaxis]

cons_conf_idx_arr = np.asarray(data['cons.conf.idx'])
cons_conf_idx_arr = (( (cons_conf_idx_arr - cons_conf_idx_arr.min()) / (cons_conf_idx_arr.max() - cons_conf_idx_arr.min()) )*2 - 1)[:,np.newaxis]

euribor3m_arr = np.asarray(data['euribor3m'])
euribor3m_arr = (( (euribor3m_arr - euribor3m_arr.min()) / (euribor3m_arr.max() - euribor3m_arr.min()) )*2 - 1)[:,np.newaxis]

nr_employed_arr = np.asarray(data['nr.employed'])
nr_employed_arr = (( (nr_employed_arr - nr_employed_arr.min()) / (nr_employed_arr.max() - nr_employed_arr.min()) )*2 - 1)[:,np.newaxis]


job_arr = np.asarray(data['job'])
marital_arr = np.asarray(data['marital'])
education_arr = np.asarray(data['education'])
default_arr = np.asarray(data['default'])
housing_arr = np.asarray(data['housing'])
loan_arr = np.asarray(data['loan'])
contact_arr = np.asarray(data['contact'])
month_arr = np.asarray(data['month'])
day_of_week_arr = np.asarray(data['day_of_week'])
poutcome_arr = np.asarray(data['poutcome'])
y_arr = np.asarray(data['y'])


data_arr = np.vstack((job_arr, marital_arr, education_arr, default_arr, housing_arr, 
	loan_arr, contact_arr, month_arr, day_of_week_arr, poutcome_arr, y_arr)).T

print(data_arr.shape)

pos_samples = data_arr[data_arr[:,-1]=='yes']
neg_samples = data_arr[data_arr[:,-1]=='no']

print(pos_samples.shape)
print(neg_samples.shape)

for i in range(len(features_cat)):
	fig, ax = plt.subplots(1,2, figsize=(9, 5), sharey=True)

	pos_labels, pos_counts = np.unique(pos_samples[:,i], return_counts=True)
	pos_counts = pos_counts / np.sum(pos_counts)
	ax[0].bar(pos_labels, pos_counts, facecolor='g', alpha=0.75)
	ax[0].set_title('Positive Samples')
	ax[0].set_xticklabels(pos_labels, rotation='vertical')
	neg_labels, neg_counts = np.unique(neg_samples[:,i], return_counts=True)
	neg_counts = neg_counts / np.sum(neg_counts)
	ax[1].bar(neg_labels, neg_counts, facecolor='r', alpha=0.75)
	ax[1].set_title('Negative Samples')
	ax[1].set_xticklabels(neg_labels, rotation='vertical')
	fig.suptitle(features_cat[i])

	fig.savefig(features_cat[i] + '.png',bbox_inches='tight')

# plt.show()
	









