import pandas as pd
from sklearn.linear_model import LogisticRegression


# Training Data - train.csv

training_features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                     'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


target = ["y"]

dataset = pd.read_csv("train.csv")


# data = pd.read_csv("train.csv", usecols = ['age','job','marital','education','default','housing','loan','contact','month',
# 	                                       'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate',
# 	                                       'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']).to_dict(orient="records")


# X_Train = pd.read_csv("train.csv", usecols = training_features).to_dict(orient="records")
training_data = pd.read_csv("train.csv", usecols=training_features)

# print(type(training_data))
# quit()

for index, row in training_data.iterrows():
    # print(row['age'])

    # Process age - biinning use weight of evidence
    # row['age'] = something

    # Process job
    job_dict = {'admin.': 1, 'self-employed': 2, 'technician': 3, 'management': 4, 'services': 5, 'student': 6,
                'unemployed': 7, 'housemaid': 8, 'blue-collar': 9, 'entrepreneur': 10, 'retired': 11, 'unknown': 12}

    if row['job'] in ['retired', 'student']:
        row['job'] = 1
    else:
        row['job'] = 0

    # Process marital status

    marital_status = {'single': 1, 'divorced': 2, 'married': 3, 'unknown': 4}

    if row['marital'] in ['single', 'unknown']:
        row['marital'] = 1
    else:
        row['marital'] = 0


    # Process Education

    education_dict = {'high.school': 1, 'university.degree': 2, 'professional.course': 3, 'unknown': 4,
                      'basic.6y': 5, 'basic.4y': 6, 'basic.9y': 7, 'illiterate': 8}

    if row['education'] in ['illiterate', 'university.degree']:
        row['education'] = 1
    else:
        row['education'] = 0

    # Process Default

    default_dict = {'unknown': 2, 'no': 0, 'yes': 1}

    if row['default'] in ['no']:
        row['default'] = 1
    else:
        row['default'] = 0


    # Process housing

    housing_dict = {'unknown': 0, 'no': 0, 'yes': 1}

    if row['housing'] in housing_dict:
        row['housing'] = housing_dict[row['housing']]

    # Process Loan

    loan_dict = {'unknown': 0, 'no': 1, 'yes': 0}

    if row['loan'] in loan_dict:
        row['loan'] = loan_dict[row['loan']]

    # Process contact

    contact_dict = {'cellular': 1, 'telephone': 0}

    if row['contact'] in contact_dict:
        row['contact'] = contact_dict[row['contact']]

    # Process Month

    month_dict = {'mar': 1, 'apr': 2, 'may': 3, 'jun': 4,
                  'jul': 5, 'aug': 6, 'sep': 7, 'oct': 8, 'nov': 9, 'dec': 10}

    if row['month'] in ['mar', 'sep', 'oct', 'dec']:
        row['month'] = 1
    else:
        row['month'] = 0


    # Process Weekday

    week_dict = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}

    if row['day_of_week'] in ['tue', 'thu']:
        row['day_of_week'] = 1
    else:
        row['day_of_week'] = 0

    print(row['day_of_week'])

    # Process Duration

    duration = 100

    if row['duration'] < duration:
        row['duration'] = 0
    else:
        row['duration'] = 1

    # Process Campaign

    camp = 25

    if row['campaign'] < camp:
        row['campaign'] = 1
    else:
        row['campaign'] = 0

    
    # Process pdays

    pdays = 999

    if row['pdays'] < pdays:
        row['pdays'] = 1
    else:
        row['pdays'] = 0


    # Process pdays

    if row['pdays'] >=3 and row['pdays'] <=6:
        row['pdays'] = 1
    else:
        row['pdays'] = 0

    # Process poutcome

    pout_dict = {'nonexistent':0, 'failure': 1, 'success':2}

    if row['poutcome'] in pout_dict:
        row['poutcome'] = pout_dict[row['poutcome']]

    # Process emp.var.rate

    if row['emp.var.rate'] <= -1.1:
        row['emp.var.rate'] = 1
    else:
        row['emp.var.rate'] = 0


    # Process cons.price.idx

    cons_price_idx = 93.749

    if row['cons.price.idx'] < cons_price_idx:
        row['cons.price.idx'] = 0
    else:
        row['cons.price.idx'] = 1


    # Process cons.conf.idx

    if row['cons.conf.idx'] > -40.8 and row['cons.conf.idx'] < -37.5:
        row['cons.conf.idx'] = 1
    else:
        row['cons.conf.idx'] = 0


    # Process euribor3m

    if row['euribor3m'] > 0.683 and row['euribor3m'] < 0.699:
        row['euribor3m'] = 1
    else:
        row['euribor3m'] = 0


    # Process nr.employed

    if row['nr.employed'] >= 4963.6 and row['nr.employed'] <= 5023.5:
        row['nr.employed'] = 1
    else:
        row['nr.employed'] = 0




# Y_Train = pd.read_csv("train.csv", usecols = target).to_dict(orient="records")
Y_Train = pd.read_csv("train.csv", usecols=target)

# Now data is a list of all client records
# print(X_Train)
# print(Y_Train)

# Test if dict is correct
# for record in X_Train:
# 	print(record)
# 	quit()


clf = LogisticRegression()
clf.fit(X_Train, Y_Train)
