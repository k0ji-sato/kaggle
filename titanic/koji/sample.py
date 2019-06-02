# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

#標準化
#--------------------------------------------------------------------
def normalization(x):
    x_ = np.array(x)
    return (x_ - x_.mean(axis=0)) / (x_.max(axis=0) - x_.min(axis=0))
#--------------------------------------------------------------------

#Dataクラスの定義
#--------------------------------------------------------------------
class Data:

## データの読み込み
## 数値データでないSexデータを0,1に置換
    def __init__(self):
        self.df_train = pd.read_csv("./rawdata/train.csv").replace("male",0).replace("female",1)
        self.df_test = pd.read_csv("./rawdata/test.csv").replace("male",0).replace("female",1)

## 関数Xの定義
## X("feature")で"feature"で指定したデータの列を標準化して呼び出し
## 返り値は学習データとテストデータの2つ
    def X(self, *args):
        featureList = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                         'Ticket', 'Fare', 'Cabin', 'Embarked']
        for key in args:
            if key not in featureList:
                raise ValueError('{0} is invalid key'.format(key))

### "Age"の欠損値は平均値で補完する
        self.df_train["Age"].fillna(self.df_train.Age.median(), inplace=True)
        self.df_test["Age"].fillna(self.df_test.Age.median(), inplace=True)

        return normalization(self.df_train[list(args)].values), normalization(self.df_test[list(args)].values)

## 関数y()の定義
## y()で正解ラベルの呼び出し
    def y(self):
        return self.df_train["Survived"].values.astype('int')

## 関数result2csvの定義
## 推定ラベルのベクトルを渡すとcsvに
    def result2csv(self, predictedLabel, filename='result.csv'):
        self.df_test["Survived"] = predictedLabel
        self.df_test[["PassengerId", "Survived"]].to_csv("./result/{0}".format(filename), index=False)


    def grade_hist(self):
        split_data = []
        for survived in [0,1]:
            split_data.append(self.df_train[self.df_train.Survived==survived])

        temp = [i["Pclass"].dropna() for i in split_data]
        plt.hist(temp, histtype="barstacked", bins=3)
#--------------------------------------------------------------------


#======================================================================
if __name__ == '__main__':
    data = Data()
    x_train, x_test = data.X('Age', 'Sex', 'Parch', 'Pclass')
    y_train = data.y()

    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    lr.fit(x_train,y_train)

    y_test = lr.predict(x_test)
    data.result2csv(y_test)
