import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv("/home/drproduck/Documents/creditcard.csv")

from sklearn import ensemble
features = ['V7', 'V10', 'V18', 'V4', 'V9', 'V16', 'V14', 'V11', 'V17', 'V12']
print(features)
allfeatures = ['V7', 'V10', 'V18', 'V4', 'V9', 'V16', 'V14', 'V11', 'V17', 'V12', 'Class']
print(allfeatures)
truncated = df[allfeatures]

truncated.groupby(df.Class).apply(len)

positive = truncated[truncated.Class==1]
negative = truncated[truncated.Class==0]
test_set = pd.concat((positive.iloc[::2, :], negative.iloc[::2, :]), axis=0)
train_set = pd.concat((positive.iloc[1::2, :], negative.iloc[1::2, :]), axis=0)

print(test_set.groupby(df.Class).apply(len))
print(train_set.groupby(df.Class).apply(len))

import seaborn as sns

for feature in df.columns:
    pl.figure()
    pl.title(feature)
    sns.distplot(df[feature][df.Class==1])
    sns.distplot(df[feature][df.Class==0])
    pl.savefig(feature)

def train_and_test_with(mlfunc_list):
    for func, name in mlfunc_list:
        func.fit(train_set[features], train_set.Class)
        print(name+'\n', pd.crosstab(test_set.Class, func.predict(test_set[features])), '\n')

from sklearn import svm
from sklearn import tree
train_and_test_with([(ensemble.RandomForestClassifier(), 'random forest'), (ensemble.AdaBoostClassifier(), 'adaptive boosting'),
                     (ensemble.BaggingClassifier(), 'bagging'), (ensemble.ExtraTreesClassifier(), 'extra tree'),
                     (ensemble.GradientBoostingClassifier(), 'gradient boosting')])

train_and_test_with([(tree.DecisionTreeClassifier(), 'decision tree'), (svm.SVC(), 'support vector machine')])

from sklearn import naive_bayes
