import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
deci = svm.SVC()
deci.fit(df[df.columns[0:4]], df.species)
pred1 = (deci.predict(iris.data))
print(pd.crosstab(df.species, deci.predict(df[df.columns[0:4]])))
deci = neighbors.KNeighborsClassifier()
deci.fit(df.iloc[:,:4], df.species)
pred2  = (deci.predict(iris.data))

diff = np.equal(pred1, pred2)
print(diff)
print(sum(1 if x else 0 for x in diff)/len(diff))

deci = ensemble.RandomForestClassifier()
deci.fit(df[iris.feature_names], df.species)

print(pd.crosstab(df.species, deci.predict(df[iris.feature_names])))

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_features="auto",
                                  min_samples_leaf=10)
clf.fit(df[iris.feature_names], df.species)

def confusion():
    print(pd.crosstab(df.species, deci.predict(df[iris.feature_names])))

confusion()


