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
# print(training_data.head())

dummy = training_data.head()
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
