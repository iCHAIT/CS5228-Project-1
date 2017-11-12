import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
from sklearn.metrics import matthews_corrcoef



# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10000,n_jobs=-1,max_depth=13,min_samples_leaf=20)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model



###############################################################
def read_data():
    data_train = pd.read_csv("_Train_0.85.csv")
    data_test = pd.read_csv("_Train_0.15.csv")


    train_y = data_train.label
    train_x = data_train.drop('label', axis=1)
    train_x = train_x.drop('id', axis=1)

    test_y = data_test.label
    test_x = data_test.drop('label', axis=1)
    test_x = test_x.drop('id', axis=1)

    return train_x, train_y, test_x, test_y

def read_data1():
    data_test = pd.read_csv("_Test_truth.csv")

    test_true_y = data_test.label
    test_true_x = data_test.drop('label', axis=1)
    test_true_x = test_true_x.drop('id', axis=1)
    return test_true_x, test_true_y



if __name__ == '__main__':

    test_classifiers = ['NB',
                        'KNN',
                        'LR',
                        'RF',
                        'DT',
                        'SVM',
                        # 'SVMCV',
                        'GBDT'
                        ]
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   # 'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y= read_data()

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))

###train_0.15#########################################################
        print("----train_0.15")
        predict1 = model.predict(test_x)




        precision1 = metrics.precision_score(test_y, predict1)
        recall1 = metrics.recall_score(test_y, predict1)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision1, 100 * recall1))
        accuracy1 = metrics.accuracy_score(test_y, predict1)
        print('accuracy: %.2f%%' % (100 * accuracy1))
        accuracy11 = matthews_corrcoef(test_y, predict1)
        print('matthews_corrcoef_accuracy: %.5f%%' % (100 * accuracy11))


        confusion_matrix = metrics.confusion_matrix(test_y, predict1)
        print(confusion_matrix)

###test_truth#########################################################
        test_true_x, test_true_y = read_data1()
        print("----test_truth")
        predict2 = model.predict(test_true_x)

        #################################
        result_name = "sampleSubmission" + classifier + ".csv"
        file_object = open(result_name, 'w')
        # file_object.write("id,prediction\n" )
        for item in range(len(predict2)):
            file_object.write("%s," % item)
            file_object.write("%s\n" % int(predict2[item]))
        file_object.close()
        ################################

        precision2 = metrics.precision_score(test_true_y, predict2)
        recall2 = metrics.recall_score(test_true_y, predict2)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision2, 100 * recall2))
        accuracy2 = metrics.accuracy_score(test_true_y, predict2)
        print('accuracy: %.2f%%' % (100 * accuracy2))
        accuracy21 = matthews_corrcoef(test_true_y, predict2)
        print('matthews_corrcoef_accuracy: %.5f%%' % (100 * accuracy21))
