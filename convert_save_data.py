import pandas as pd
import os
import numpy as np

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

data_file = os.path.join(data_dir, 'train.csv')
train_data = pd.read_csv(data_file, usecols=features)
num_train = len(train_data)

data_file = os.path.join(data_dir, 'test.csv')
test_data = pd.read_csv(data_file, usecols=features)
num_test = len(test_data)

data = pd.concat([train_data,test_data])


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


data_arr = np.hstack((age_arr, duration_arr, campaign_arr, pdays_arr, previous_arr, emp_var_rate_arr, cons_price_idx_arr, 
	cons_conf_idx_arr, euribor3m_arr, nr_employed_arr, job_arr, marital_arr, education_arr, default_arr, housing_arr, 
	loan_arr, contact_arr, month_arr, day_of_week_arr, poutcome_arr))

X_train = data_arr[:num_train,:]
X_test = data_arr[num_train:,:]

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


out_file = os.path.join(data_dir, 'X_test.npy')
np.save(out_file, X_test)


num_train_samples = X_train.shape[0]
indeces = np.arange(num_train_samples)
np.random.shuffle(indeces)
num_folds = 5
num_samples_in_fold = (np.floor(num_train_samples/num_folds)).astype(int)
print('Number of Samples in a Fold: ' + str(num_samples_in_fold))

for i in range(num_folds):
	mask = np.zeros(num_train_samples, dtype=bool)
	indeces_range = indeces[i*num_samples_in_fold:(i+1)*num_samples_in_fold]
	mask[indeces_range]= True
	temp_X_validation = X_train[mask,:]
	temp_Y_validation = Y_train[mask,:]
	mask = np.invert(mask)
	temp_X_train = X_train[mask,:]
	temp_Y_train = Y_train[mask,:]

	print('####################################')
	print('Validation data size in fold' + str(i) + ': ' + str(temp_X_validation.shape))
	print('Validation label size in fold' + str(i) + ': ' + str(temp_Y_validation.shape))
	print('Train data size in fold' + str(i) + ': ' + str(temp_X_train.shape))
	print('Train label size in fold' + str(i) + ': ' + str(temp_Y_train.shape))

	out_file = os.path.join(data_dir, 'X_train_fold' + str(i) + '.npy')
	np.save(out_file, temp_X_train)

	out_file = os.path.join(data_dir, 'Y_train_fold' + str(i) + '.npy')
	np.save(out_file, temp_Y_train)

	out_file = os.path.join(data_dir, 'X_validation_fold' + str(i) + '.npy')
	np.save(out_file, temp_X_validation)

	out_file = os.path.join(data_dir, 'Y_validation_fold' + str(i) + '.npy')
	np.save(out_file, temp_Y_validation)
	









