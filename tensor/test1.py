import collections
import random

# a = collections.Counter(x=1, y=2)
# print(a.get('x'))
# a['x'] += 11
# print(a['x'])
# a.popitem()
# print(a)
#
# for _ in range(10):
#     print(random.randint(1,2))


# a = [1,2,3]
# b = [[100,2],[2,3],[3,4]]
c = [1.0,2.0,3.,4.,5.]
# import tensorflow as tf
# with tf.Session() as sess:
#     #print(sess.run(tf.nn.weighted_cross_entropy_with_logits(targets=[0.0, 1.0], logits=[-100.0, 500], pos_weight=1.0)))
#     print(tf.nn.dropout(x=c, keep_prob=1).eval())

import numpy as np
df = np.array([[1,1],[2,1],[1,0]])
#print(np.where(df[:,1] == 1))
fraud = df[np.where(df[:,1:2] == 1),:]
normal = df[np.where(df[:,1:2] == 0),:]


#print(c[2: 10])

#print(fraud)

s = set('')
s = {x for x in c}
print(len(s))
print(s)
a = set()
a.add(1), a.add(2), a.add(3)
print(a)
c = [1.0 if x == 1.0 else 0.0 for x in c]
print(c)

