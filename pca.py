import pandas as pd
import csv
import os
import numpy as np
from sklearn.decomposition import PCA


# Training Data - train.csv

training_features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                     'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


target = ["y"]

job_dict = {'admin.': 1, 'self-employed': 2, 'technician': 3, 'management': 4, 'services': 5, 'student': 6,
                'unemployed': 7, 'housemaid': 8, 'blue-collar': 9, 'entrepreneur': 10, 'retired': 11, 'unknown': 12}

training_data = pd.read_csv("train.csv", usecols=training_features)

pout_dict = {'nonexistent':0, 'failure': 1, 'success':2}

marital_status = {'single': 1, 'divorced': 2, 'married': 3, 'unknown': 0}

education_dict = {'high.school': 1, 'university.degree': 2, 'professional.course': 3, 'unknown': 0,
                  'basic.6y': 4, 'basic.4y': 5, 'basic.9y': 6, 'illiterate': 7}

default_dict = {'unknown': 0, 'no': 1, 'yes': 2}

housing_dict = {'unknown': 0, 'no': 1, 'yes': 2}

loan_dict = {'unknown': 0, 'no': 1, 'yes': 2}

contact_dict = {'cellular': 1, 'telephone': 0}

month_dict = {'mar': 1, 'apr': 2, 'may': 3, 'jun': 4,
              'jul': 5, 'aug': 6, 'sep': 7, 'oct': 8, 'nov': 9, 'dec': 10}

week_dict = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}

X_Train = []


for index, row in training_data.iterrows():

    if row['job'] in job_dict:
        row['job'] = job_dict[row['job']]
    else:
        row['job'] = 0

    # Process marital status

    if row['marital'] in marital_status:
        row['marital'] = marital_status[row['marital']]
    else:
        row['marital'] = 0


    # Process Education

    if row['education'] in education_dict:
        row['education'] = education_dict[row['education']]
    else:
        row['education'] = 0

    # Process Default


    if row['default'] in default_dict:
        row['default'] = default_dict[row['default']]
    else:
        row['default'] = 0


    # Process housing

    if row['housing'] in housing_dict:
        row['housing'] = housing_dict[row['housing']]
    else:
        row['housing'] = 0

    # Process Loan

    if row['loan'] in loan_dict:
        row['loan'] = loan_dict[row['loan']]
    else:
        row['loan'] = 0

    # Process contact

    if row['contact'] in contact_dict:
        row['contact'] = contact_dict[row['contact']]
    else:
        row['contact'] = 0

    # Process Month

    if row['month'] in month_dict:
        row['month'] = month_dict[row['month']]
    else:
        row['month'] = 0


    # Process Weekday

    if row['day_of_week'] in week_dict:
        row['day_of_week'] = week_dict[row['day_of_week']]
    else:
        row['day_of_week'] = 0


    if row['poutcome'] in pout_dict:
        row['poutcome'] = pout_dict[row['poutcome']]
    else:
        row['poutcome'] = 0

    X_Train.append([row['age'], row['job'], row['marital'], row['education'], row['default'], row['housing'], row['loan'], row['contact'], 
                    row['month'], row['day_of_week'], row['duration'], row['campaign'], row['pdays'], row['previous'], row['poutcome'],
                     row['emp.var.rate'], row['cons.price.idx'], row['cons.conf.idx'], row['euribor3m'], row['nr.employed']])


X_Train = (X_Train - np.mean(X_Train,axis=0,keepdims=True))/np.std(X_Train,axis=0,keepdims=True)


pca = PCA()
X_Train_fit = pca.fit_transform(X_Train)
print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
