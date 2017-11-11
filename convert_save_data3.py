import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA

data_dir = '/Users/mustafauo/Dropbox/NUS_Academic/NUS_2017_2018_1/CS5228/Banking_Project/Python_Code/'

features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
job_dict = {
'admin.': [1,0,0,0,0,0,0,0,0,0,0], 
'self-employed': [0,1,0,0,0,0,0,0,0,0,0], 
'technician': [0,0,1,0,0,0,0,0,0,0,0], 
'management': [0,0,0,1,0,0,0,0,0,0,0], 
'services': [0,0,0,0,1,0,0,0,0,0,0], 
'student': [0,0,0,0,0,1,0,0,0,0,0], 
'unemployed': [0,0,0,0,0,0,1,0,0,0,0], 
'housemaid': [0,0,0,0,0,0,0,1,0,0,0], 
'blue-collar': [0,0,0,0,0,0,0,0,1,0,0], 
'entrepreneur': [0,0,0,0,0,0,0,0,0,1,0], 
'retired': [0,0,0,0,0,0,0,0,0,0,1], 
'unknown': [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]}

marital_dict = {
'single': [1,0,0], 
'divorced': [0,1,0], 
'married': [0,0,1], 
'unknown': [-1,-1,-1]}

education_dict = {
'high.school': [1,0,0,0,0,0,0], 
'university.degree': [0,1,0,0,0,0,0], 
'professional.course': [0,0,1,0,0,0,0], 
'unknown': [0,0,0,1,0,0,0],
'basic.6y': [0,0,0,0,1,0,0], 
'basic.4y': [0,0,0,0,0,1,0], 
'basic.9y': [0,0,0,0,0,0,1], 
'illiterate': [-1,-1,-1,-1,-1,-1,-1]}

default_dict = {
'no': [1,0], 
'yes': [0,1],
'unknown': [-1,-1]}

housing_dict = {
'no': [1,0], 
'yes': [0,1],
'unknown': [-1,-1]}

loan_dict = {
'no': [1,0], 
'yes': [0,1],
'unknown': [-1,-1]}

contact_dict = {
'cellular': [-1], 
'telephone': [1]}

month_dict = {
'jan': [1,0,0,0,0,0,0,0,0,0,0], 
'feb': [0,1,0,0,0,0,0,0,0,0,0], 
'mar': [0,0,1,0,0,0,0,0,0,0,0], 
'apr': [0,0,0,1,0,0,0,0,0,0,0], 
'may': [0,0,0,0,1,0,0,0,0,0,0], 
'jun': [0,0,0,0,0,1,0,0,0,0,0], 
'jul': [0,0,0,0,0,0,1,0,0,0,0], 
'aug': [0,0,0,0,0,0,0,1,0,0,0], 
'sep': [0,0,0,0,0,0,0,0,1,0,0], 
'oct': [0,0,0,0,0,0,0,0,0,1,0], 
'nov': [0,0,0,0,0,0,0,0,0,0,1], 
'dec': [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]}

day_of_week_dict = {
'mon': [1,0,0,0,0,0], 
'tue': [0,1,0,0,0,0], 
'wed': [0,0,1,0,0,0], 
'thu': [0,0,0,1,0,0],
'fri': [0,0,0,0,1,0], 
'sat': [0,0,0,0,0,1],  
'sun': [-1,-1,-1,-1,-1,-1]}

poutcome_dict = {
'failure': [1,0], 
'success': [0,1],
'nonexistent': [-1,-1]}

def normalize(arr):
	# return (( (arr - arr.min()) / (arr.max() - arr.min()) )*2-1)[:,np.newaxis]
	return (( (arr - arr.mean()) / (arr.std()) ))[:,np.newaxis]

data_file = os.path.join(data_dir, 'train.csv')
train_data = pd.read_csv(data_file, usecols=features)
num_train = len(train_data)

data_file = os.path.join(data_dir, 'test.csv')
test_data = pd.read_csv(data_file, usecols=features)
num_test = len(test_data)

data = pd.concat([train_data,test_data])


age_arr = np.asarray(data['age'])
age_arr = normalize(age_arr)

duration_arr = np.asarray(data['duration'])
duration_arr = normalize(duration_arr)

campaign_arr = np.asarray(data['campaign'])
campaign_arr = normalize(campaign_arr)

pdays_arr = np.asarray(data['pdays'])
ind = np.asarray(pdays_arr == 999)
pdays_mean = np.mean(pdays_arr[np.invert(ind)])
pdays_arr[ind] = pdays_mean
pdays_arr = normalize(pdays_arr)

previous_arr = np.asarray(data['previous'])
previous_arr = normalize(previous_arr)

emp_var_rate_arr = np.asarray(data['emp.var.rate'])
emp_var_rate_arr = normalize(emp_var_rate_arr)

cons_price_idx_arr = np.asarray(data['cons.price.idx'])
cons_price_idx_arr = normalize(cons_price_idx_arr)

cons_conf_idx_arr = np.asarray(data['cons.conf.idx'])
cons_conf_idx_arr = normalize(cons_conf_idx_arr)

euribor3m_arr = np.asarray(data['euribor3m'])
euribor3m_arr = normalize(euribor3m_arr)

nr_employed_arr = np.asarray(data['nr.employed'])
nr_employed_arr = normalize(nr_employed_arr)


job_arr = []
marital_arr = []
education_arr = []
default_arr = []
housing_arr = []
loan_arr = []
contact_arr = []
month_arr = []
day_of_week_arr = []
poutcome_arr = []

for index, row in data.iterrows():
	job_arr.append(job_dict[row['job']])
	marital_arr.append(marital_dict[row['marital']])
	education_arr.append(education_dict[row['education']])
	default_arr.append(default_dict[row['default']])
	housing_arr.append(housing_dict[row['housing']])
	loan_arr.append(loan_dict[row['loan']])
	contact_arr.append(contact_dict[row['contact']])
	month_arr.append(month_dict[row['month']])
	day_of_week_arr.append(day_of_week_dict[row['day_of_week']])
	poutcome_arr.append(poutcome_dict[row['poutcome']])

job_arr = np.asarray(job_arr)
marital_arr = np.asarray(marital_arr)
education_arr = np.asarray(education_arr)
default_arr = np.asarray(default_arr)
housing_arr = np.asarray(housing_arr)
loan_arr = np.asarray(loan_arr)
contact_arr = np.asarray(contact_arr)
month_arr = np.asarray(month_arr)
day_of_week_arr = np.asarray(day_of_week_arr)
poutcome_arr = np.asarray(poutcome_arr)

num_data_arr = np.hstack((age_arr, duration_arr, campaign_arr, pdays_arr, previous_arr, emp_var_rate_arr, cons_price_idx_arr, 
	cons_conf_idx_arr, euribor3m_arr, nr_employed_arr))

# pca = PCA(n_components=6, svd_solver='full')
# num_data_arr = pca.fit_transform(num_data_arr)
# # pca.fit(num_data_arr)
# cum_sum = np.cumsum(pca.explained_variance_ratio_)
# print(cum_sum)

print(num_data_arr.shape)
print(job_arr.shape)

data_arr = np.hstack((num_data_arr, job_arr, marital_arr, education_arr, default_arr, housing_arr, 
	loan_arr, contact_arr, month_arr, day_of_week_arr, poutcome_arr))

# data_arr = (( (data_arr - data_arr.min(axis=0)) / (data_arr.max(axis=0) - data_arr.min(axis=0)) )*2 - 1)


X_train = data_arr[:num_train,:]
X_test = data_arr[num_train:,:]

print(X_test.shape)

out_file = os.path.join(data_dir, 'X_test_num_pca.npy')
np.save(out_file, X_test)

print('Test data size: ' + str(X_test.shape))

target = ['y']

target_dict = {
'no': [1, 0],
'yes': [0, 1]
}

data_file = os.path.join(data_dir, 'train.csv')
label_data = pd.read_csv(data_file, usecols=target)

label_arr = []
for index, row in label_data.iterrows():
	label_arr.append(target_dict[row['y']])

Y_train = np.asarray(label_arr)

ind_pos = np.where(Y_train[:,1] == 1)[0]
ind_neg = np.where(Y_train[:,1] == 0)[0]

X_train_pos = X_train[ind_pos,:]
X_train_neg = X_train[ind_neg,:]

Y_train_pos = Y_train[ind_pos,:]
Y_train_neg = Y_train[ind_neg,:]

print(X_train_pos.shape)
print(X_train_neg.shape)

num_pos_samples = ind_pos.size
num_neg_samples = ind_neg.size

num_train_samples = X_train.shape[0]

indices_pos = np.arange(num_pos_samples)
np.random.shuffle(indices_pos)

indices_neg = np.arange(num_neg_samples)
np.random.shuffle(indices_neg)

num_folds = 5
num_pos_samples_in_fold = (np.floor(num_pos_samples/num_folds)).astype(int)
num_neg_samples_in_fold = (np.floor(num_neg_samples/num_folds)).astype(int)
print('Number of Samples in a Fold: ' + str(num_pos_samples_in_fold + num_neg_samples_in_fold))

for i in range(num_folds):
	mask_pos = np.zeros(num_pos_samples, dtype=bool)
	mask_pos[i*num_pos_samples_in_fold:(i+1)*num_pos_samples_in_fold] = True
	print(np.sum(mask_pos))
	temp_X_pos_validation = X_train_pos[mask_pos,:]
	temp_Y_pos_validation = Y_train_pos[mask_pos,:]
	mask_pos = np.invert(mask_pos)
	temp_X_pos_train = X_train_pos[mask_pos,:]
	temp_Y_pos_train = Y_train_pos[mask_pos,:]

	mask_neg = np.zeros(num_neg_samples, dtype=bool)
	mask_neg[i*num_neg_samples_in_fold:(i+1)*num_neg_samples_in_fold]= True
	print(np.sum(mask_neg))
	temp_X_neg_validation = X_train_neg[mask_neg,:]
	temp_Y_neg_validation = Y_train_neg[mask_neg,:]
	mask_neg = np.invert(mask_neg)
	temp_X_neg_train = X_train_neg[mask_neg,:]
	temp_Y_neg_train = Y_train_neg[mask_neg,:]

	temp_X_validation = np.vstack((temp_X_pos_validation,temp_X_neg_validation))
	temp_Y_validation = np.vstack((temp_Y_pos_validation,temp_Y_neg_validation))
	temp_X_train = np.vstack((temp_X_pos_train,temp_X_neg_train))
	temp_Y_train = np.vstack((temp_Y_pos_train,temp_Y_neg_train))

	print('####################################')
	print('Validation data size in fold' + str(i) + ': ' + str(temp_X_validation.shape))
	print('Validation label size in fold' + str(i) + ': ' + str(temp_Y_validation.shape))
	print('Train data size in fold' + str(i) + ': ' + str(temp_X_train.shape))
	print('Train label size in fold' + str(i) + ': ' + str(temp_Y_train.shape))

	out_file = os.path.join(data_dir, 'X_train_num_pca_fold' + str(i) + '.npy')
	np.save(out_file, temp_X_train)

	out_file = os.path.join(data_dir, 'Y_train_num_pca_fold' + str(i) + '.npy')
	np.save(out_file, temp_Y_train)

	out_file = os.path.join(data_dir, 'X_validation_num_pca_fold' + str(i) + '.npy')
	np.save(out_file, temp_X_validation)

	out_file = os.path.join(data_dir, 'Y_validation_num_pca_fold' + str(i) + '.npy')
	np.save(out_file, temp_Y_validation)

	csv_filename = os.path.join(data_dir, 'Y_validation_num_pca_fold' + str(i) + '.txt')
	np.savetxt(csv_filename, temp_Y_validation, delimiter=',')
	









