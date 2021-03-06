# coding=utf-8
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/home/drproduck/Downloads/ibm.csv')
for c in data.columns:
    s = {x for x in data[c]}
    print(c, len(s))
data.Attrition = [1 if x == 'Yes' else 0 for x in data.Attrition]
print(data.columns)
data.drop(['Over18', 'EmployeeCount', 'StandardHours'], axis=1, inplace=True)
for c in data.columns:
    if all(isinstance(n, str) for n in data[c]):
        s = {x for x in data[c]}
        t = list(s)
        dic = {}
        for z in range(len(s)):
            dic[t[z]] = z
        print(dic)
        data[c] = [dic[x] for x in data[c]]

plt.figure(1)
plt.hist([data.BusinessTravel[data.Attrition==1],data.BusinessTravel[data.Attrition==0]], color=['green','red'], label=['quit', 'stay'])

matrix = data.as_matrix()

attrition = matrix[:,1:2]
attri_shape = np.shape(attrition)
onehot = np.zeros((attri_shape[0], attri_shape[1]*2))
for i in range(len(attrition)):
    if attrition[i] == 1:
        onehot[i][0] = 1
    else: onehot[i][1] = 1
matrix = np.append(matrix, onehot, axis=1)
matrix = np.delete(matrix, 1, axis=1)

print(np.shape(matrix))
import collections
c = collections.Counter()
for x in matrix[:,31]:
    c[x] += 1
print(c)
c = collections.Counter()
for x in matrix[:,32]:
    c[x] += 1
print(c)
num_features = 31
# matrix[:, :num_features] = (matrix[:,:num_features] - matrix[:,:num_features].mean(axis=0))/matrix[:,num_features].var(axis=0)
np.save('matrix', matrix)
np.savetxt('matrix', matrix)
print(matrix[:1,:])





