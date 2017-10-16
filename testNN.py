import pandas as pd
from sklearn.linear_model import LogisticRegression
import csv
import os
import numpy as np


# Training Data - train.csv

training_features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                     'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


target = ["y"]

dataset = pd.read_csv("test.csv")

training_data = pd.read_csv("test.csv", usecols=training_features)


X_Train = []


age_arr = np.asarray(dataset['age'])
age_std = np.std(age_arr)
age_mean = np.mean(age_arr)


duration_arr = np.asarray(dataset['duration'])
duration_std = np.std(duration_arr)
duration_mean = np.mean(duration_arr)


campaign_arr = np.asarray(dataset['campaign'])
campaign_std = np.std(campaign_arr)
campaign_mean = np.mean(campaign_arr)


pdays_arr = np.asarray(dataset['pdays'])
pdays_std = np.std(pdays_arr)
pdays_mean = np.mean(pdays_arr)


previous_arr = np.asarray(dataset['previous'])
previous_std = np.std(previous_arr)
previous_mean = np.mean(previous_arr)


emp_var_arr = np.asarray(dataset['emp.var.rate'])
emp_var_std = np.std(emp_var_arr)
emp_var_mean = np.mean(emp_var_arr)

cons_price_arr = np.asarray(dataset['cons.price.idx'])
cons_price_std = np.std(cons_price_arr)
cons_price_mean = np.mean(cons_price_arr)

cons_conf_arr = np.asarray(dataset['cons.conf.idx'])
cons_conf_std = np.std(cons_conf_arr)
cons_conf_mean = np.mean(cons_conf_arr)


euribor3m_arr = np.asarray(dataset['euribor3m'])
euribor3m_std = np.std(euribor3m_arr)
euribor3m_mean = np.mean(euribor3m_arr)

nremployed_arr = np.asarray(dataset['nr.employed'])
nremployed_std = np.std(nremployed_arr)
nremployed_mean = np.mean(nremployed_arr)

for index, row in training_data.iterrows():

    row['age'] = (row['age'] - age_mean)/age_std

    # Process job
    job_dict = {'admin.': 1, 'self-employed': 2, 'technician': 3, 'management': 4, 'services': 5, 'student': 6,
                'unemployed': 7, 'housemaid': 8, 'blue-collar': 9, 'entrepreneur': 10, 'retired': 11, 'unknown': 0}

    if row['job'] in job_dict:
        row['job'] = job_dict[row['job']]
    else:
        row['job'] = 0

    # Process marital status

    marital_status = {'single': 1, 'divorced': 2, 'married': 3, 'unknown': 0}

    if row['marital'] in marital_status:
        row['marital'] = marital_status[row['marital']]
    else:
        row['marital'] = 0


    # Process Education

    education_dict = {'high.school': 1, 'university.degree': 2, 'professional.course': 3, 'unknown': 0,
                      'basic.6y': 4, 'basic.4y': 5, 'basic.9y': 6, 'illiterate': 7}

    if row['education'] in education_dict:
        row['education'] = education_dict[row['education']]
    else:
        row['education'] = 0

    # Process Default

    default_dict = {'unknown': 0, 'no': 1, 'yes': 2}

    if row['default'] in default_dict:
        row['default'] = default_dict[row['default']]
    else:
        row['default'] = 0


    # Process housing

    housing_dict = {'unknown': 0, 'no': 1, 'yes': 2}

    if row['housing'] in housing_dict:
        row['housing'] = housing_dict[row['housing']]
    else:
    	row['housing'] = 0

    # Process Loan

    loan_dict = {'unknown': 0, 'no': 1, 'yes': 2}

    if row['loan'] in loan_dict:
        row['loan'] = loan_dict[row['loan']]
    else:
    	row['loan'] = 0

    # Process contact

    contact_dict = {'cellular': 1, 'telephone': 0}

    if row['contact'] in contact_dict:
        row['contact'] = contact_dict[row['contact']]
    else:
    	row['contact'] = 0

    # Process Month

    month_dict = {'mar': 1, 'apr': 2, 'may': 3, 'jun': 4,
                  'jul': 5, 'aug': 6, 'sep': 7, 'oct': 8, 'nov': 9, 'dec': 10}

    if row['month'] in month_dict:
        row['month'] = month_dict[row['month']]
    else:
        row['month'] = 0


    # Process Weekday

    week_dict = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}

    if row['day_of_week'] in week_dict:
        row['day_of_week'] = week_dict[row['day_of_week']]
    else:
        row['day_of_week'] = 0


    # Process Duration

    row['duration'] = (row['duration'] - duration_mean)/duration_std

    # Process Campaign

    row['campaign'] = (row['campaign'] - campaign_mean)/campaign_std    
    
    # Process pdays

    row['pdays'] = (row['pdays'] - pdays_mean)/pdays_std


    # Process previous

    row['previous'] = (row['previous'] - previous_mean)/previous_std

    # Process poutcome

    pout_dict = {'nonexistent':0, 'failure': 1, 'success':2}

    if row['poutcome'] in pout_dict:
        row['poutcome'] = pout_dict[row['poutcome']]
    else:
    	row['poutcome'] = 0

    # Process emp.var.rate

    row['emp.var.rate'] = (row['emp.var.rate'] - emp_var_mean)/emp_var_std

    # Process cons.price.idx

    row['cons.price.idx'] = (row['cons.price.idx'] - cons_price_mean)/cons_price_std


    # Process cons.conf.idx

    row['cons.conf.idx'] = (row['cons.conf.idx'] - cons_conf_mean)/cons_conf_std


    # Process euribor3m

    row['euribor3m'] = (row['euribor3m'] - euribor3m_mean)/euribor3m_std


    # Process nr.employed

    row['nr.employed'] = (row['nr.employed'] - nremployed_mean)/nremployed_std


    X_Train.append([row['age'], row['job'], row['marital'], row['education'], row['default'], row['housing'], row['loan'], row['contact'], 
                    row['month'], row['day_of_week'], row['duration'], row['campaign'], row['pdays'], row['previous'], row['poutcome'],
                     row['emp.var.rate'], row['cons.price.idx'], row['cons.conf.idx'], row['euribor3m'], row['nr.employed'] ])


    print(X_Train)
    quit()

