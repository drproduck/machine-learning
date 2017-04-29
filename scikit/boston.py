from sklearn import datasets
boston = datasets.load_boston()

import pandas as pd
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['price'] = boston.target

from sklearn.linear_model import LinearRegression

features = ['AGE', 'LSTAT', 'TAX']
# lm = LinearRegression()
# lm.fit(df[features], df.price)

print(df.tail(5))

import numpy as np
import pylab as pl
# add your actual vs. predicted points
# pl.scatter(df.price, lm.predict(df[features]))
# # add the line of perfect fit
# straight_line = np.arange(0, 60)
# pl.plot(straight_line)
# pl.title("Fitted Values")
#
# """apply, groupby, split continue, merge, join, agg (aggregate)"""
#
# a = np.reshape(range(1,10), (3,3))
# print(a)
# subset = df[['AGE', 'TAX', 'price', 'LSTAT']]
# print(subset.groupby(['AGE', 'price'], sort=False).apply(lambda x: 0))
#
# # pl.show()

from sklearn import ensemble

# deci = ensemble.RandomForestClassifier()
# deci.fit(df[df.columns], df.price)
# importances = deci.feature_importances_
# sorted_idx = np.argsort(importances)
#
# padding = np.arange(len(features)) + 0.5
# pl.barh(padding, importances[sorted_idx], align='center')
# pl.yticks(padding, features[sorted_idx])
# pl.xlabel("Relative Importance")
# pl.title("Variable Importance")
# pl.show()
