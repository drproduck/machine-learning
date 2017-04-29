import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv("/home/drproduck/Downloads/creditcard.csv")
print(df.tail())


grouped_taget = df.groupby(by='Class', sort=False)
print(grouped_taget.agg([np.min, np.max]))


from sklearn import ensemble

ju = ensemble.RandomForestClassifier()
ju.fit(df.iloc[:, 0:30], df.Class)
importances = ju.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(30) + 0.5
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, df.columns[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")

# import seaborn as sns
#
# sns.distplot(df.Amount[df.Class == 1], bins=50)
# sns.distplot(df.Amount[df.Class== 0], bins=50)
pl.show()

df = df[['V14', 'V12', 'V16', '']]