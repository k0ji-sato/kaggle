#%%
import platform
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd

def normalization(x):
    x_ = np.array(x)
    return (x_ - x_.mean(axis=0)) / (x_.max(axis=0) - x_.min(axis=0))

class Data:
    def __init__(self):
        self.df_train = pd.read_csv("./rawdata/train.csv").replace("male",0).replace("female",1)
        self.df_test = pd.read_csv("./rawdata/test.csv").replace("male",0).replace("female",1)

    def X(self, *args):
        featureList = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                         'Ticket', 'Fare', 'Cabin', 'Embarked']
        for key in args:
            if key not in featureList:
                raise ValueError('{0} is invalid key'.format(key))

        self.df_train["Age"].fillna(self.df_train.Age.median(), inplace=True)
        self.df_test["Age"].fillna(self.df_test.Age.median(), inplace=True)

        return normalization(self.df_train[list(args)].values), normalization(self.df_test[list(args)].values)

    def y(self):
        return self.df_train["Survived"].values.astype('int')

    def result2csv(self, predictedLabel, filename='result.csv'):
        self.df_test["Survived"] = predictedLabel
        self.df_test[["PassengerId", "Survived"]].to_csv("./result/{0}".format(filename), index=False)

    def grade_hist(self):
        split_data = []
        for survived in [0,1]:
            split_data.append(self.df_train[self.df_train.Survived==survived])

        temp = [i["Pclass"].dropna() for i in split_data]
        plt.hist(temp, histtype="barstacked", bins=3)

data = Data()
#data.grade_hist()
X_train, X_test = data.X('Age', 'Sex', 'Parch', 'Pclass')
y_train = data.y()
#print(normalization(X))

#%%
tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]
classifier =  GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy', verbose=True, n_jobs=2)
classifier.fit(X_train, y_train)
print(classifier.cv_results_)

#%%
estimator = classifier.best_estimator_
#estimator = svm.SVC(C=1, kernel='linear')
#estimator.fit(X_train, y_train)

y_test = estimator.predict(X_test)
data.result2csv(y_test)

