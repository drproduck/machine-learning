import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

#df = pd.read_csv("/home/drproduck/Downloads/creditcard.csv")
#print(df.loc[df.Class==1])

normal = np.load('normal.npy')*10
np.random.shuffle(normal)
train_normal = normal[::2, :]
test_normal = normal[1::2, :]
fraud = np.load('fraud.npy')*10

mu = train_normal[:, :19].mean(axis=0)

sigma = train_normal[:, :19].std(axis=0)

#dist = tf.contrib.distributions.Normal(mu, sigma)
sess = tf.InteractiveSession()



def distribution(dataset, color):
    dist = tf.contrib.distributions.Normal(mu[0], sigma[0])
    pdf_train_normal = dist.pdf(np.float64(dataset[:, 0:1])).eval()
    for i in range(1, 19):
        dist = tf.contrib.distributions.Normal(mu[i], sigma[i])
        x = dist.pdf(np.float64(dataset[:, i:i + 1])).eval()
        pdf_train_normal = np.concatenate((pdf_train_normal, x), axis=1)

    score = np.prod(pdf_train_normal, axis=1)

    print(score.max(), score.min())

    sns.distplot(score, bins=50, color=color)

distribution(train_normal, 'green')
distribution(fraud, 'red')
#normal = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))

# for i in range(len(test_normal)):
#     for j in range(30):
#          test_normal[i][j] = np.sqrt(1/(2*np.pi*sigma[j] ** 2)) * np.exp(-(test_normal[i][j] - mu[j]) ** 2 / (2 * sigma[j] ** 2))
#
# train_normal_distribution = np.prod(train_normal, axis=1)
# print(train_normal_distribution)

# j = 0
# for i in df.columns[2:5]:
#     print(i)
#     plt.figure(j)
#     sns.distplot(df[i][df.Class == 1], bins =50, label='fraud')
#     sns.distplot(df[i][df.Class == 0], bins = 50, label='normal')
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     j += 1

# plt.hist(df.V1[df.Class == 1], bins =50, label='fraud', alpha=0.5)
# plt.hist(df.V2[df.Class == 1], bins = 50, label='normal', alpha=0.5)
# df.iloc[:,:30] = (df.iloc[:, :30] - df.iloc[:, :30].mean(axis=0)) / df.iloc[:, :30].var(axis=0)
#df = (df.columns - df.columns.mean)/df.columns.var
#plt.hist(df.V5[df.Class == 0], bins =50)

# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])



plt.show()