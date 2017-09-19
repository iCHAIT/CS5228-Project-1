import pandas as pd
from sklearn.linear_model import LogisticRegression


# Training Data - train.csv

training_features = ['age','job','marital','education','default','housing','loan','contact','month',
	                                       'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate',
	                                       'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']


target = ["y"]

dataset = pd.read_csv("train.csv")


# data = pd.read_csv("train.csv", usecols = ['age','job','marital','education','default','housing','loan','contact','month',
# 	                                       'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate',
# 	                                       'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']).to_dict(orient="records")





# X_Train = pd.read_csv("train.csv", usecols = training_features).to_dict(orient="records")
training_data = pd.read_csv("train.csv", usecols = training_features)

# print(type(training_data))
# quit()

for index, row in training_data.iterrows():
	print(row['age'])

	# Process age - biinning use weight of evidence
	# row['age'] = something

	# Process job
	job_dict = {'admin.':1,'self-employed':2,'technician':3, 'management':4,'services':5,'student':6,
	            'unemployed':7,'housemaid':8,'blue-collar':9,'entrepreneur':10,'retired':11,'unknown':12}

	if row['job'] in job_dict:
		row['job'] = job_dict[row['job']]

	# print(row['job'])

	# Process marital status
	marital_status = {'single': 1, 'divorced': 2, 'married': 3, 'unknown': 4}

	if row['marital'] in marital_status:
		row['marital'] = marital_status[row['marital']]

	# print(row['marital'])


	# Process Education

	education_dict = {'high.school':1,'university.degree':2,'professional.course':3,'unknown':4,
                      'basic.6y':5,'basic.4y':6,'basic.9y':7,'illiterate':8}

	if row['education'] in education_dict:
		row['education'] = education_dict[row['education']]              

	# Process Default

	default_dict = {'unknown': 2, 'no': 0, 'yes': 1}

	if row['default'] in default_dict:
		row['default'] = default_dict[row['default']]

	# print(row['default'])


	# Process housing

	housing_dict = {'unknown': 2, 'no': 0, 'yes': 1}

	if row['housing'] in housing_dict:
		row['housing'] = housing_dict[row['housing']]

	# print(row['housing'])

	# Process Loan

	housing_dict = {'unknown': 2, 'no': 0, 'yes': 1}

	if row['loan'] in housing_dict:
		row['loan'] = housing_dict[row['loan']]

	# print(row['loan'])


	# Process contact

	contact_dict = {'cellular': 1, 'telephone': 2}

	if row['contact'] in contact_dict:
		row['contact'] = contact_dict[row['contact']]


	print(row['contact'])


	# Process Month

	month_dict = {'mar':1,'apr':2, 'may':3, 'jun':4, 'jul':5, 'aug':6, 'sep':7, 'oct':8, 'nov':9, 'dec':10}

	if row['month'] in month_dict:
		row['month'] = month_dict[row['month']]	

	print(row['month'])

















	quit()

print(dummy)


# Y_Train = pd.read_csv("train.csv", usecols = target).to_dict(orient="records")
Y_Train = pd.read_csv("train.csv", usecols = target)

# Now data is a list of all client records
# print(X_Train)
# print(Y_Train)

# Test if dict is correct
# for record in X_Train:
# 	print(record)
# 	quit()




clf = LogisticRegression()
clf.fit(X_Train, Y_Train)
